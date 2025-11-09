"""Time series analysis for trend visualization.

Detects time columns and generates time-series aggregations for KPI trends.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from discovery.analytics.relationships import detect_foreign_key_patterns


def detect_primary_time_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely primary time column for a table.
    
    Returns the first datetime/timestamp column by ordinal order (column position).
    Also attempts to convert object columns to datetime if they contain date/time data.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Column name of primary time column, or None if none found
    """
    # Find all datetime columns in ordinal order
    time_cols = []
    for col in df.columns:
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_cols.append(col)
        # Try converting object columns to datetime
        elif df[col].dtype == 'object':
            try:
                # Try to convert to datetime
                converted = pd.to_datetime(df[col], errors='coerce')
                # Check if conversion was successful (at least some valid dates)
                if converted.notna().sum() > 0:
                    # If successful, convert the column in the dataframe
                    df[col] = converted
                    time_cols.append(col)
            except Exception:
                # Conversion failed, skip this column
                pass
    
    if not time_cols:
        return None
    
    # Return the first datetime column by ordinal order
    return time_cols[0]


def aggregate_kpi_over_time(
    df: pd.DataFrame,
    time_column: str,
    metric_column: str,
    aggregation: str = "sum",
    time_granularity: str = "day",
) -> pd.DataFrame:
    """Aggregate a metric column over time.
    
    Args:
        df: DataFrame with time and metric columns
        time_column: Name of time column
        metric_column: Name of metric column
        aggregation: Aggregation function ("sum", "avg", "count", "count_distinct", "min", "max")
        time_granularity: Time granularity ("day", "week", "month", "year")
    
    Returns:
        DataFrame with time and aggregated metric
    """
    if time_column not in df.columns or metric_column not in df.columns:
        return pd.DataFrame()
    
    # Ensure time column is datetime
    df_time = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_time[time_column]):
        # Try converting object columns to datetime
        if df_time[time_column].dtype == 'object':
            df_time[time_column] = pd.to_datetime(df_time[time_column], errors="coerce")
        else:
            df_time[time_column] = pd.to_datetime(df_time[time_column], errors="coerce")
    
    if metric_column not in df_time.columns:
        return pd.DataFrame()
    
    # Handle count_distinct separately (don't convert metric to numeric)
    if aggregation == "count_distinct":
        # Skip if metric column is datetime (can't count distinct datetime as meaningful metric)
        if pd.api.types.is_datetime64_any_dtype(df_time[metric_column]):
            return pd.DataFrame()
        
        # Remove rows with null time only (keep all metric values for distinct count)
        df_time = df_time[[time_column, metric_column]].dropna(subset=[time_column])
        
        if len(df_time) == 0:
            return pd.DataFrame()
        
        # Map granularity to pandas frequency
        freq_map = {
            "hour": "H",
            "day": "D",
            "week": "W",
            "month": "M",
            "year": "Y",
        }
        freq = freq_map.get(time_granularity, "D")
        
        # Group by time period and count distinct values
        df_time['period'] = df_time[time_column].dt.to_period(freq)
        distinct_counts = df_time.groupby('period')[metric_column].nunique()
        
        # Convert period index back to datetime and create Series
        period_timestamps = distinct_counts.index.to_timestamp()
        aggregated = pd.Series(distinct_counts.values, index=period_timestamps)
        aggregated.index.name = time_column
    else:
        # For other aggregations, ensure metric column is numeric
        # Skip if metric column is datetime (can't aggregate datetime as metric)
        if pd.api.types.is_datetime64_any_dtype(df_time[metric_column]):
            return pd.DataFrame()
        
        # Convert metric column to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(df_time[metric_column]):
            # Handle boolean dtype: true=1, false=0, null=0
            if df_time[metric_column].dtype == 'bool':
                df_time[metric_column] = df_time[metric_column].fillna(False).astype(int)
            else:
                df_time[metric_column] = pd.to_numeric(df_time[metric_column], errors="coerce")
        
        # Remove rows with null time or metric
        df_time = df_time[[time_column, metric_column]].dropna()
        
        if len(df_time) == 0:
            return pd.DataFrame()
        
        # Set time as index and resample
        df_time = df_time.set_index(time_column)
        
        # Map granularity to pandas frequency
        freq_map = {
            "hour": "H",
            "day": "D",
            "week": "W",
            "month": "M",
            "year": "Y",
        }
        freq = freq_map.get(time_granularity, "D")
        
        # Ensure we're only aggregating the metric column (not the index)
        metric_series = df_time[metric_column]
        
        # Aggregate
        if aggregation == "sum":
            aggregated = metric_series.resample(freq).sum()
        elif aggregation == "avg":
            aggregated = metric_series.resample(freq).mean()
        elif aggregation == "count":
            aggregated = metric_series.resample(freq).count()
        elif aggregation == "min":
            aggregated = metric_series.resample(freq).min()
        elif aggregation == "max":
            aggregated = metric_series.resample(freq).max()
        else:
            aggregated = metric_series.resample(freq).sum()
    
    # Reset index to get time as column
    result = aggregated.reset_index()
    result.columns = [time_column, metric_column]
    
    return result


def select_optimal_granularity(
    df: pd.DataFrame,
    time_column: str,
) -> str:
    """Select optimal time granularity based on data characteristics.
    
    Args:
        df: DataFrame with time column
        time_column: Name of time column
    
    Returns:
        Optimal granularity: "day", "week", "month", "year", or "hour"
    """
    if time_column not in df.columns:
        return "day"
    
    time_series = df[time_column].dropna()
    if len(time_series) == 0:
        return "day"
    
    # Ensure datetime (try converting object columns)
    if not pd.api.types.is_datetime64_any_dtype(time_series):
        if time_series.dtype == 'object':
            converted = pd.to_datetime(time_series, errors="coerce").dropna()
            if len(converted) > 0:
                # Update the dataframe column if conversion successful
                df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
                time_series = converted
            else:
                time_series = converted
        else:
            time_series = pd.to_datetime(time_series, errors="coerce").dropna()
    
    if len(time_series) < 2:
        return "day"
    
    time_range_days = (time_series.max() - time_series.min()).days
    time_range_hours = (time_series.max() - time_series.min()).total_seconds() / 3600
    num_points = len(time_series)
    
    # If very short range (< 1 day) and many points, use hour
    if time_range_hours < 24 and num_points > 10:
        return "hour"
    
    # If range is less than 7 days, use day
    if time_range_days < 7:
        return "day"
    
    # If range is less than 90 days, use day or week based on density
    if time_range_days < 90:
        points_per_day = num_points / max(time_range_days, 1)
        if points_per_day < 0.5:  # Sparse data
            return "week"
        return "day"
    
    # If range is less than 365 days, use week or month
    if time_range_days < 365:
        points_per_day = num_points / max(time_range_days, 1)
        if points_per_day < 0.1:  # Very sparse data
            return "month"
        return "week"
    
    # For longer ranges, use month or year
    if time_range_days < 365 * 3:
        return "month"
    
    return "year"


def get_best_kpi_for_column(
    column_key: str,
    kpi_suggestions: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Get the single best KPI for a column based on relevance and type.
    
    Prioritizes KPIs that match the column's numeric type (measure vs dimension).
    
    Args:
        column_key: Column identifier (e.g., "table.column")
        kpi_suggestions: List of KPI suggestion dictionaries
        profile: Optional column profile to check numeric type
    
    Returns:
        Best KPI dictionary or None
    """
    matching_kpis = get_top_kpis_for_column(column_key, kpi_suggestions, top_k=10)
    
    if not matching_kpis:
        return None
    
    # If we have profile info, prioritize KPIs that match the column type
    numeric_type = profile.get("numericType") if profile else None
    
    # Score each KPI
    scored_kpis = []
    for kpi in matching_kpis:
        score = kpi.get("relevance_score", 0.5)
        kpi_type = kpi.get("kpi_type", "")
        
        # Boost score for measures if column is a measure
        if numeric_type == "measure":
            if kpi_type in ["sum", "average", "trend"]:
                score *= 1.2
        # Boost score for dimensions if column is a dimension
        elif numeric_type == "dimension":
            if kpi_type in ["count", "percentage"]:
                score *= 1.2
        
        # Prefer trend KPIs for time series
        if kpi_type == "trend":
            score *= 1.1
        
        scored_kpis.append((kpi, score))
    
    # Sort by score and return best
    scored_kpis.sort(key=lambda x: x[1], reverse=True)
    return scored_kpis[0][0] if scored_kpis else None


def get_top_kpis_for_column(
    column_key: str,
    kpi_suggestions: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Get top KPI suggestions for a specific column.
    
    Args:
        column_key: Column identifier (e.g., "table.column")
        kpi_suggestions: List of KPI suggestion dictionaries
        top_k: Number of top KPIs to return
    
    Returns:
        List of top KPI suggestions sorted by relevance
    """
    table, column = column_key.split(".", 1) if "." in column_key else ("", column_key)
    
    matching_kpis = []
    
    for kpi in kpi_suggestions:
        kpi_col = kpi.get("column", "")
        kpi_cols = kpi.get("columns", [])
        kpi_table = kpi.get("table", "")
        
        # Match by column name (with or without table prefix)
        matches = False
        
        # Direct match
        if kpi_col == column or kpi_col == column_key:
            matches = True
        # Match in columns list
        elif isinstance(kpi_cols, list) and column in kpi_cols:
            matches = True
        # Match by table and column
        elif kpi_table == table and (kpi_col == column or column in kpi_cols):
            matches = True
        # Match metric_column for trend KPIs
        elif kpi.get("metric_column") == column or kpi.get("metric_column") == column_key:
            matches = True
        
        if matches:
            matching_kpis.append(kpi)
    
    # Sort by relevance score
    matching_kpis.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    return matching_kpis[:top_k]


def prepare_trend_data(
    table_samples: Dict[str, pd.DataFrame],
    priority_columns: List[str],
    kpi_suggestions: List[Dict[str, Any]],
    profiles: Dict[str, Any],
    top_kpis_per_column: int = 1,
) -> Dict[str, Any]:
    """Prepare trend data for high-priority columns.
    
    Shows the best KPI for each column with optimal time granularity.
    Includes all columns that have time/date data, not just measures/dimensions.
    
    Args:
        table_samples: Dictionary mapping table names to DataFrames
        priority_columns: List of column keys (e.g., ["table1.col1"])
        kpi_suggestions: List of KPI suggestion dictionaries
        profiles: Dictionary mapping column keys to profiles
        top_kpis_per_column: Number of KPIs to show per column (default 1 for best only)
    
    Returns:
        Dictionary with 'trends' (successful) and 'failed' (with reasons) lists
    """
    trends = []
    failed_trends = []
    
    for column_key in priority_columns:
        if "." not in column_key:
            failed_trends.append({
                "column_key": column_key,
                "reason": "Invalid column key format (missing table prefix)"
            })
            continue
        
        table, column = column_key.split(".", 1)
        
        if table not in table_samples:
            failed_trends.append({
                "column_key": column_key,
                "reason": f"Table '{table}' not found in samples"
            })
            continue
        
        df = table_samples[table]
        
        if column not in df.columns:
            failed_trends.append({
                "column_key": column_key,
                "reason": f"Column '{column}' not found in table '{table}'"
            })
            continue
        
        # Get column profile
        profile = profiles.get(column_key, {})
        numeric_type = profile.get("numericType")
        
        # Detect primary time column for this table
        time_column = detect_primary_time_column(df)
        if not time_column:
            failed_trends.append({
                "column_key": column_key,
                "reason": f"No datetime/timestamp columns found in table '{table}'"
            })
            continue
        
        # Get best KPI for this column (or use column itself if no KPI)
        best_kpi = get_best_kpi_for_column(column_key, kpi_suggestions, profile)
        
        # Check if ID/code column
        is_id_code = detect_foreign_key_patterns(column)
        
        # If no KPI found, try to use the column directly if it's numeric or boolean
        if not best_kpi:
            if pd.api.types.is_numeric_dtype(df[column]) and df[column].dtype != 'bool':
                # For ID/code columns, use count_distinct
                if is_id_code:
                    best_kpi = {
                        "kpi_type": "count_distinct",
                        "description": f"Distinct count of {column}",
                        "formula": f"COUNT(DISTINCT {column})",
                        "column": column,
                        "relevance_score": 0.5,
                    }
                else:
                    # Create a simple KPI for the column itself
                    best_kpi = {
                        "kpi_type": "sum",
                        "description": f"Trend of {column}",
                        "formula": f"SUM({column})",
                        "column": column,
                        "relevance_score": 0.5,
                    }
            elif df[column].dtype == 'bool':
                # For boolean columns, create a sum KPI (true=1, false=0, null=0)
                best_kpi = {
                    "kpi_type": "sum",
                    "description": f"Count of {column} (true values)",
                    "formula": f"SUM({column})",
                    "column": column,
                    "relevance_score": 0.7,
                }
            else:
                failed_trends.append({
                    "column_key": column_key,
                    "reason": f"No KPI suggestions found and column is not numeric or boolean"
                })
                continue
        
        # Try different ways to get the metric column
        metric_col = best_kpi.get("metric_column") or best_kpi.get("column") or column
        
        if metric_col not in df.columns:
            # Try without table prefix if column_key had one
            metric_col_short = metric_col.split(".")[-1] if "." in metric_col else metric_col
            if metric_col_short in df.columns:
                metric_col = metric_col_short
            else:
                failed_trends.append({
                    "column_key": column_key,
                    "reason": f"Metric column '{metric_col}' not found in table"
                })
                continue
        
        # Skip if metric column is datetime (can't aggregate datetime as metric)
        if pd.api.types.is_datetime64_any_dtype(df[metric_col]):
            failed_trends.append({
                "column_key": column_key,
                "reason": f"Metric column '{metric_col}' is datetime type (cannot aggregate)"
            })
            continue
        
        # Skip if metric column is the same as time column
        if metric_col == time_column:
            failed_trends.append({
                "column_key": column_key,
                "reason": f"Metric column '{metric_col}' is the same as time column '{time_column}'"
            })
            continue
        
        # For ID/code columns, skip numeric conversion (count_distinct works on any type)
        # For other aggregations, ensure metric column is numeric
        if not is_id_code:
            if not pd.api.types.is_numeric_dtype(df[metric_col]):
                if df[metric_col].dtype == 'bool':
                    # Convert boolean to int: true=1, false=0, null=0
                    df[metric_col] = df[metric_col].fillna(False).astype(int)
                else:
                    # Try to convert to numeric
                    try:
                        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
                        if df[metric_col].isna().all():
                            failed_trends.append({
                                "column_key": column_key,
                                "reason": f"Metric column '{metric_col}' cannot be converted to numeric"
                            })
                            continue
                    except Exception:
                        failed_trends.append({
                            "column_key": column_key,
                            "reason": f"Metric column '{metric_col}' is not numeric and cannot be converted"
                        })
                        continue
        
        # Determine aggregation based on KPI type
        # For ID/code columns, use count_distinct
        if is_id_code:
            aggregation = "count_distinct"
        else:
            kpi_type = best_kpi.get("kpi_type", "sum")
            # For boolean columns, always use sum
            if df[metric_col].dtype == 'bool' or (hasattr(df[metric_col], 'dtype') and df[metric_col].dtype == 'int64' and df[metric_col].isin([0, 1]).all()):
                aggregation = "sum"
            else:
                aggregation_map = {
                    "sum": "sum",
                    "average": "avg",
                    "count": "count",
                    "trend": "sum",
                    "ratio": "avg",
                    "percentage": "avg",
                }
                aggregation = aggregation_map.get(kpi_type, "sum")
        
        # Select optimal granularity based on data characteristics
        granularity = select_optimal_granularity(df, time_column)
        
        # Aggregate over time
        try:
            trend_df = aggregate_kpi_over_time(
                df, time_column, metric_col, aggregation, granularity
            )
            
            if len(trend_df) == 0:
                failed_trends.append({
                    "column_key": column_key,
                    "reason": f"Time aggregation returned empty result (no valid data points)"
                })
                continue
            
            # Convert to dict format, ensuring datetime is serialized properly
            # Use date_format='iso' to preserve datetime information
            data_records = []
            for _, row in trend_df.iterrows():
                record = {}
                for col in trend_df.columns:
                    val = row[col]
                    # Convert datetime to ISO string for JSON serialization
                    if pd.api.types.is_datetime64_any_dtype(trend_df[col]):
                        record[col] = val.isoformat() if pd.notna(val) else None
                    else:
                        record[col] = val if pd.notna(val) else None
                data_records.append(record)
            
            trends.append({
                "column_key": column_key,
                "table": table,
                "column": column,
                "numeric_type": numeric_type,
                "time_column": time_column,
                "metric_column": metric_col,
                "kpi": best_kpi,
                "aggregation": aggregation,
                "granularity": granularity,
                "data": data_records,
            })
        except Exception as e:
            failed_trends.append({
                "column_key": column_key,
                "reason": f"Error during time aggregation: {str(e)}"
            })
            continue
    
    return {
        "trends": trends,
        "failed": failed_trends,
    }

