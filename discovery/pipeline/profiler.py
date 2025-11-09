from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd


def _classify_numeric_type(series: pd.Series, distinct: int, total_rows: int) -> str:
    """Classify numeric column as 'dimension' or 'measure'.
    
    Dimensions are typically:
    - Low cardinality (< 20% of rows)
    - Integer values (or very few decimal places)
    - Used for grouping/filtering
    
    Measures are typically:
    - High cardinality (> 50% of rows)
    - Continuous values with decimals
    - Used for aggregation (sum, avg, etc.)
    
    Args:
        series: Numeric pandas Series
        distinct: Number of distinct values
        total_rows: Total number of rows
        
    Returns:
        'dimension' or 'measure'
    """
    if total_rows == 0 or distinct == 0:
        return "measure"
    
    # Convert to numeric, handling errors
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "measure"
    
    # Calculate cardinality ratio
    cardinality_ratio = distinct / total_rows
    
    # Check if values are mostly integers
    is_integer_dtype = pd.api.types.is_integer_dtype(series)
    
    # Check if values have decimal places (for non-integer dtypes)
    has_decimals = False
    if not is_integer_dtype:
        # Sample values to check for decimals
        sample = s.head(1000)
        has_decimals = any((sample % 1) != 0)
    
    # Check value range characteristics
    value_range = s.max() - s.min()
    std_dev = s.std()
    cv = std_dev / s.mean() if s.mean() != 0 else float('inf')  # Coefficient of variation
    
    # Classification logic
    # Low cardinality (< 20%) suggests dimension
    if cardinality_ratio < 0.2:
        # If integer-like and low cardinality, likely dimension
        if is_integer_dtype or not has_decimals:
            return "dimension"
        # Even with decimals, very low cardinality might be dimension (e.g., rating scales)
        if distinct <= 10:
            return "dimension"
    
    # Very high cardinality (> 80%) strongly suggests measure
    if cardinality_ratio > 0.8:
        return "measure"
    
    # Medium cardinality: use additional heuristics
    # Integer columns with medium cardinality could be dimensions (e.g., age, year)
    if is_integer_dtype and cardinality_ratio < 0.5:
        # Check if values are sequential/consecutive (common in dimensions)
        sorted_unique = sorted(s.unique()[:100])  # Sample for performance
        if len(sorted_unique) > 1:
            diffs = np.diff(sorted_unique)
            # If mostly consecutive integers, likely dimension
            if np.allclose(diffs, 1.0, atol=0.1):
                return "dimension"
    
    # High coefficient of variation suggests measure (wide spread)
    if cv > 1.0 and cardinality_ratio > 0.3:
        return "measure"
    
    # Default: if high cardinality or has decimals, treat as measure
    if cardinality_ratio > 0.5 or has_decimals:
        return "measure"
    
    # Default to measure for ambiguous cases
    return "measure"


def _numeric_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute robust numeric summary statistics with quantiles.

    Coerces values to numeric to handle mixed-type columns gracefully.
    Excludes boolean dtype to avoid boolean arithmetic errors.
    """
    # Explicitly exclude boolean dtype
    if series.dtype == 'bool':
        return {}
    
    # Convert to numeric, handling boolean-like values
    s = pd.to_numeric(series, errors="coerce")
    # Ensure we don't have boolean dtype after conversion
    if s.dtype == 'bool':
        s = s.astype(int)
    if s.dropna().empty:
        return {}
    q = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "min": float(np.nanmin(s)),
        "max": float(np.nanmax(s)),
        "mean": float(np.nanmean(s)),
        "std": float(np.nanstd(s, ddof=1)) if s.count() > 1 else 0.0,
        "p1": float(q.loc[0.01]),
        "p5": float(q.loc[0.05]),
        "p50": float(q.loc[0.5]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
    }


def profile_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Profile each column: dtype, null rate, distinct count, and type-specific stats."""
    rows = len(df)
    profiles: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = float(null_count / rows) if rows else 0.0
        distinct = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        col_profile: Dict[str, Any] = {
            "dtype": dtype,
            "nullPct": null_pct,
            "distinct": distinct,
        }
        if pd.api.types.is_numeric_dtype(series) and series.dtype != 'bool':
            col_profile["stats"] = _numeric_stats(series)
            col_profile["numericType"] = _classify_numeric_type(series, distinct, rows)
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_profile["min"] = str(series.min())
            col_profile["max"] = str(series.max())
        else:
            vc = series.astype("string").value_counts(dropna=True).head(10)
            col_profile["topK"] = [{"value": str(i), "count": int(c)} for i, c in vc.items()]
        profiles[col] = col_profile
    return profiles


def profile_table(df: pd.DataFrame) -> Dict[str, Any]:
    """Profile table-level stats such as row count and duplicate percentage."""
    duplicate_pct = float(1.0 - len(df.drop_duplicates()) / len(df)) if len(df) else 0.0
    return {
        "rowCount": int(len(df)),
        "duplicatePct": duplicate_pct,
    }


