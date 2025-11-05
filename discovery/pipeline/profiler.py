from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd


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


