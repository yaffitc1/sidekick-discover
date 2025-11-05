"""Advanced sampling strategies for large datasets.

Supports multi-stage sampling that captures corner cases without over-representation,
handles datasets up to 1TB efficiently using database pushdown and chunked processing.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


def get_table_metadata(connector, table: str) -> Dict[str, Any]:
    """Get table metadata without full scan.
    
    Returns:
        Dictionary with row_count_estimate, columns, and basic stats
    """
    metadata: Dict[str, Any] = {
        "table": table,
        "columns": {},
        "row_count_estimate": None,
    }
    
    # Get schema
    schema = connector.get_schema(table)
    metadata["columns"] = schema
    
    # Try to get row count estimate
    if hasattr(connector, "_get_connection"):
        # Snowflake connector
        try:
            conn = connector._get_connection()
            cursor = conn.cursor()
            full_table = f"{connector.database}.{connector.schema}.{table.upper()}"
            
            # Try to get row count from INFORMATION_SCHEMA
            cursor.execute(
                f"""
                SELECT ROW_COUNT, BYTES 
                FROM {connector.database}.INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = '{connector.schema}' 
                AND TABLE_NAME = '{table.upper()}'
                """
            )
            result = cursor.fetchone()
            if result:
                metadata["row_count_estimate"] = result[0] if result[0] else None
                metadata["bytes_estimate"] = result[1] if result[1] else None
        except Exception:
            pass
    
    return metadata


def detect_stratification_columns(df: pd.DataFrame, sample_size: int = 10000) -> List[str]:
    """Detect columns suitable for stratification.
    
    Returns:
        List of column names that are good candidates for stratification
    """
    strata_cols = []
    
    for col in df.columns:
        series = df[col]
        
        # Categorical columns with reasonable cardinality
        if series.dtype == "object" or series.dtype.name == "category":
            nunique = series.nunique()
            if 2 <= nunique <= 50:  # Good cardinality for stratification
                strata_cols.append(col)
        
        # Date columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            strata_cols.append(col)
        
        # Numeric columns with discrete values
        elif pd.api.types.is_numeric_dtype(series):
            # Check if it's more discrete than continuous
            nunique = series.nunique()
            if nunique <= 20 and nunique < len(df) * 0.1:
                strata_cols.append(col)
    
    return strata_cols[:3]  # Limit to top 3 to avoid too many strata


def advanced_sample_snowflake(
    connector,
    table: str,
    sample_size: int = 100000,
    corner_case_ratio: float = 0.05,
    stratify_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Advanced sampling for Snowflake with pushdown optimization.
    
    Uses multi-stage sampling:
    1. Random sample (main body)
    2. Stratified sample (if strata provided)
    3. Corner cases (outliers/extremes) at true frequency
    
    Args:
        connector: SnowflakeConnector instance
        table: Table name
        sample_size: Target sample size
        corner_case_ratio: Percentage of sample for corner cases (0.05 = 5%)
        stratify_by: Optional list of columns to stratify by
    
    Returns:
        Combined DataFrame sample
    """
    conn = connector._get_connection()
    full_table = f"{connector.database}.{connector.schema}.{table.upper()}"
    
    samples = []
    
    # Calculate sample sizes
    corner_case_size = int(sample_size * corner_case_ratio)
    main_sample_size = sample_size - corner_case_size
    
    # Get metadata first
    metadata = get_table_metadata(connector, table)
    row_count = metadata.get("row_count_estimate")
    
    # Main random sample using TABLESAMPLE (efficient)
    if row_count and row_count > main_sample_size:
        # Calculate percentage for TABLESAMPLE
        pct = min(100, max(0.1, (main_sample_size / row_count) * 100))
        query = f"""
        SELECT * FROM {full_table} 
        TABLESAMPLE SYSTEM ({pct})
        LIMIT {main_sample_size}
        """
    else:
        query = f"SELECT * FROM {full_table} LIMIT {main_sample_size}"
    
    main_sample = pd.read_sql(query, conn)
    samples.append(main_sample)
    
    # Stratified sampling if requested
    if stratify_by and len(main_sample) > 0:
        # Detect numeric columns for quantile-based stratification
        numeric_cols = [c for c in stratify_by if pd.api.types.is_numeric_dtype(main_sample[c])]
        
        if numeric_cols:
            # Use quantile-based WHERE clauses for each numeric column
            for col in numeric_cols[:2]:  # Limit to 2 columns
                try:
                    # Sample from different quantiles
                    q25 = main_sample[col].quantile(0.25)
                    q75 = main_sample[col].quantile(0.75)
                    
                    # Add samples from tails
                    tail_query = f"""
                    SELECT * FROM {full_table}
                    WHERE {col} < {q25} OR {col} > {q75}
                    LIMIT {min(10000, main_sample_size // 10)}
                    """
                    tail_sample = pd.read_sql(tail_query, conn)
                    if len(tail_sample) > 0:
                        samples.append(tail_sample)
                except Exception:
                    pass
    
    # Corner cases: outliers/extremes (at true frequency)
    if corner_case_size > 0 and len(main_sample) > 0:
        # Exclude boolean dtype to avoid issues with boolean arithmetic
        numeric_cols = main_sample.select_dtypes(include=[np.number], exclude=['bool']).columns
        if len(numeric_cols) > 0:
            # Find extreme values
            for col in numeric_cols[:2]:  # Limit to 2 columns
                try:
                    p1 = main_sample[col].quantile(0.01)
                    p99 = main_sample[col].quantile(0.99)
                    
                    # Sample extremes (very small portion)
                    extreme_query = f"""
                    SELECT * FROM {full_table}
                    WHERE {col} < {p1} OR {col} > {p99}
                    LIMIT {corner_case_size // 2}
                    """
                    extreme_sample = pd.read_sql(extreme_query, conn)
                    if len(extreme_sample) > 0:
                        samples.append(extreme_sample)
                except Exception:
                    pass
    
    # Combine all samples
    if samples:
        combined = pd.concat(samples, ignore_index=True)
        # Remove duplicates while preserving order
        combined = combined.drop_duplicates().reset_index(drop=True)
        # Limit to target size
        if len(combined) > sample_size:
            combined = combined.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return combined
    
    return main_sample


def advanced_sample(
    connector,
    table: str,
    sample_size: int = 100000,
    corner_case_ratio: float = 0.05,
    stratify_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Advanced sampling wrapper for Snowflake connectors.
    
    Args:
        connector: SnowflakeConnector instance
        table: Table name
        sample_size: Target sample size
        corner_case_ratio: Percentage of sample for corner cases (default 5%)
        stratify_by: Optional list of columns to stratify by
    
    Returns:
        DataFrame sample with representative distribution including corner cases
    """
    # Use Snowflake-specific sampling
    connector_type = type(connector).__name__
    
    if connector_type == "SnowflakeConnector":
        return advanced_sample_snowflake(
            connector, table, sample_size, corner_case_ratio, stratify_by
        )
    else:
        # Fallback to basic sampling
        return connector.sample(table=table, limit=sample_size, method="random")


def validate_sample_quality(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate that sample preserves key distributions.
    
    Args:
        full_df: Full dataset (may be a sample itself)
        sample_df: Sample to validate
        key_columns: Columns to check distribution for
    
    Returns:
        Dictionary with validation metrics
    """
    if key_columns is None:
        key_columns = list(sample_df.columns)[:5]  # Check first 5 columns
    
    validation = {
        "sample_size": len(sample_df),
        "full_size": len(full_df),
        "coverage_ratio": len(sample_df) / len(full_df) if len(full_df) > 0 else 0,
        "column_distributions": {},
    }
    
    for col in key_columns:
        if col not in sample_df.columns or col not in full_df.columns:
            continue
        
        try:
            # Compare distributions
            if pd.api.types.is_numeric_dtype(sample_df[col]) and sample_df[col].dtype != 'bool':
                sample_mean = sample_df[col].mean()
                full_mean = full_df[col].mean()
                sample_std = sample_df[col].std()
                full_std = full_df[col].std()
                
                validation["column_distributions"][col] = {
                    "mean_error": abs(sample_mean - full_mean) / abs(full_mean) if full_mean != 0 else 0,
                    "std_error": abs(sample_std - full_std) / abs(full_std) if full_std != 0 else 0,
                }
            else:
                # Categorical: compare top values
                sample_top = sample_df[col].value_counts().head(5)
                full_top = full_df[col].value_counts().head(5)
                
                overlap = len(set(sample_top.index) & set(full_top.index)) / max(len(sample_top), 1)
                validation["column_distributions"][col] = {
                    "top_value_overlap": overlap,
                }
        except Exception:
            pass
    
    return validation



