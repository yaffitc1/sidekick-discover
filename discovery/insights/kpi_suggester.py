"""KPI suggestion engine for data discovery.

Analyzes data patterns to suggest relevant KPIs with calculation formulas.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def detect_time_series_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that appear to be time-series (date/datetime).
    
    Returns:
        List of column names that are datetime types
    """
    time_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_cols.append(col)
    return time_cols


def detect_numeric_metrics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect numeric columns that could be metrics.
    
    Returns:
        List of metric suggestions with KPI types
    """
    metrics = []
    # Exclude boolean dtype to avoid issues with boolean arithmetic
    numeric_cols = df.select_dtypes(include=[np.number], exclude=['bool']).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        suggestions = []
        
        # Check if suitable for sum
        if series.min() >= 0 and series.max() > 0:
            suggestions.append({
                "kpi_type": "sum",
                "formula": f"SUM({col})",
                "description": f"Total {col}",
                "relevance_score": 0.8,
            })
        
        # Check if suitable for average
        if series.std() > 0:
            suggestions.append({
                "kpi_type": "average",
                "formula": f"AVG({col})",
                "description": f"Average {col}",
                "relevance_score": 0.7,
            })
        
        # Check if suitable for count
        if series.min() >= 0:
            suggestions.append({
                "kpi_type": "count",
                "formula": f"COUNT({col})",
                "description": f"Count of {col}",
                "relevance_score": 0.6,
            })
        
        # Check for ratio potential (if values are percentages or proportions)
        if series.min() >= 0 and series.max() <= 1:
            suggestions.append({
                "kpi_type": "percentage",
                "formula": f"AVG({col}) * 100",
                "description": f"Average {col} (%)",
                "relevance_score": 0.9,
            })
        
        if suggestions:
            metrics.append({
                "column": col,
                "suggestions": suggestions,
            })
    
    return metrics


def detect_categorical_dimensions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect categorical columns suitable for grouping/segmentation.
    
    Returns:
        List of dimension suggestions
    """
    dimensions = []
    
    for col in df.columns:
        series = df[col]
        
        # Good dimensions: categorical with reasonable cardinality
        if series.dtype == "object" or series.dtype.name == "category":
            nunique = series.nunique()
            total = len(series)
            
            # Good cardinality for grouping: 2-50 distinct values
            if 2 <= nunique <= 50:
                dimensions.append({
                    "column": col,
                    "cardinality": nunique,
                    "relevance_score": 0.8 if nunique <= 20 else 0.6,
                    "description": f"Group by {col} ({nunique} distinct values)",
                })
    
    return dimensions


def suggest_composite_kpis(
    df: pd.DataFrame,
    metrics: List[Dict[str, Any]],
    dimensions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Suggest composite KPIs (ratios, rates, percentages).
    
    Args:
        df: DataFrame to analyze
        metrics: List of detected metrics
        dimensions: List of detected dimensions
    
    Returns:
        List of composite KPI suggestions
    """
    composite_kpis = []
    
    # Exclude boolean dtype to avoid issues with boolean arithmetic
    numeric_cols = df.select_dtypes(include=[np.number], exclude=['bool']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Suggest ratios between numeric columns
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                # Check if ratio makes sense
                series_a = df[col_a].dropna()
                series_b = df[col_b].dropna()
                
                if len(series_a) == 0 or len(series_b) == 0:
                    continue
                
                # Avoid division by zero or negative ratios
                if series_b.min() > 0:
                    composite_kpis.append({
                        "kpi_type": "ratio",
                        "formula": f"{col_a} / {col_b}",
                        "description": f"Ratio of {col_a} to {col_b}",
                        "relevance_score": 0.7,
                        "columns": [col_a, col_b],
                    })
                
                # Suggest percentage if one is subset of another
                if series_a.max() <= series_b.max() and series_a.min() >= 0:
                    composite_kpis.append({
                        "kpi_type": "percentage",
                        "formula": f"({col_a} / {col_b}) * 100",
                        "description": f"{col_a} as percentage of {col_b}",
                        "relevance_score": 0.8,
                        "columns": [col_a, col_b],
                    })
    
    return composite_kpis


def suggest_trend_kpis(
    df: pd.DataFrame,
    time_column: Optional[str] = None,
    metrics: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Suggest time-series trend KPIs.
    
    Args:
        df: DataFrame to analyze
        time_column: Optional datetime column name
        metrics: Optional list of detected metrics
    
    Returns:
        List of trend KPI suggestions
    """
    trend_kpis = []
    
    if time_column is None:
        time_cols = detect_time_series_columns(df)
        if not time_cols:
            return trend_kpis
        time_column = time_cols[0]
    
    if time_column not in df.columns:
        return trend_kpis
    
    # Exclude boolean dtype to avoid issues with boolean arithmetic
    numeric_cols = df.select_dtypes(include=[np.number], exclude=['bool']).columns
    
    for col in numeric_cols:
        if col == time_column:
            continue
        
        trend_kpis.append({
            "kpi_type": "trend",
            "formula": f"Trend of {col} over time",
            "description": f"Time series trend of {col}",
            "time_column": time_column,
            "metric_column": col,
            "relevance_score": 0.8,
            "columns": [time_column, col],
        })
    
    return trend_kpis


def generate_kpi_suggestions(
    df: pd.DataFrame,
    table_name: str = "unknown",
) -> Dict[str, Any]:
    """Generate comprehensive KPI suggestions for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        table_name: Name of the table
    
    Returns:
        Dictionary with all KPI suggestions organized by type
    """
    kpi_report = {
        "table": table_name,
        "simple_metrics": [],
        "dimensions": [],
        "composite_kpis": [],
        "trend_kpis": [],
        "summary": {
            "total_suggestions": 0,
            "high_relevance": 0,
            "medium_relevance": 0,
            "low_relevance": 0,
        },
    }
    
    # Detect simple metrics
    metrics = detect_numeric_metrics(df)
    kpi_report["simple_metrics"] = metrics
    
    # Detect dimensions
    dimensions = detect_categorical_dimensions(df)
    kpi_report["dimensions"] = dimensions
    
    # Suggest composite KPIs
    composite_kpis = suggest_composite_kpis(df, metrics, dimensions)
    kpi_report["composite_kpis"] = composite_kpis
    
    # Suggest trend KPIs
    trend_kpis = suggest_trend_kpis(df)
    kpi_report["trend_kpis"] = trend_kpis
    
    # Calculate summary
    all_kpis = []
    
    # Add simple metrics
    for metric in metrics:
        for suggestion in metric.get("suggestions", []):
            all_kpis.append(suggestion)
    
    # Add composite KPIs
    all_kpis.extend(composite_kpis)
    
    # Add trend KPIs
    all_kpis.extend(trend_kpis)
    
    # Count by relevance
    for kpi in all_kpis:
        score = kpi.get("relevance_score", 0)
        if score >= 0.8:
            kpi_report["summary"]["high_relevance"] += 1
        elif score >= 0.5:
            kpi_report["summary"]["medium_relevance"] += 1
        else:
            kpi_report["summary"]["low_relevance"] += 1
    
    kpi_report["summary"]["total_suggestions"] = len(all_kpis)
    
    # Sort by relevance
    if composite_kpis:
        kpi_report["composite_kpis"] = sorted(
            composite_kpis,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
    
    return kpi_report





