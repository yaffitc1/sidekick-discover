"""Column priority scoring for relationship and trend analysis.

Scores columns based on multiple factors to identify highest priority columns
for relationship graphs and trend analysis.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd


def score_column_priority(
    column_key: str,
    profile: Dict[str, Any],
    insights: List[Dict[str, Any]],
    kpi_suggestions: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> float:
    """Score a column's priority based on multiple factors.
    
    Scoring factors:
    - Data quality (lower null percentage = higher score)
    - Insight relevance (columns mentioned in insights)
    - KPI relevance (columns used in KPI suggestions)
    - Relationship importance (columns involved in relationships)
    - Numeric type (measures get higher score than dimensions)
    
    Args:
        column_key: Column identifier (e.g., "table.column")
        profile: Column profile dictionary
        insights: List of insight dictionaries
        kpi_suggestions: List of KPI suggestion dictionaries
        relationships: List of relationship dictionaries
    
    Returns:
        Priority score (0.0 to 1.0, higher is better)
    """
    score = 0.0
    
    # Factor 1: Data quality (0-0.3 points)
    null_pct = float(profile.get("nullPct", 0.0))
    quality_score = (1.0 - null_pct) * 0.3
    score += quality_score
    
    # Factor 2: Insight relevance (0-0.2 points)
    insight_score = 0.0
    for insight in insights:
        affected_cols = insight.get("affectedColumns", [])
        if column_key in affected_cols:
            insight_score += 0.1
    insight_score = min(insight_score, 0.2)  # Cap at 0.2
    score += insight_score
    
    # Factor 3: KPI relevance (0-0.25 points)
    kpi_score = 0.0
    for kpi in kpi_suggestions:
        kpi_col = kpi.get("column")
        kpi_cols = kpi.get("columns", [])
        if kpi_col == column_key or column_key in kpi_cols:
            relevance = kpi.get("relevance_score", 0.5)
            kpi_score += relevance * 0.05
    kpi_score = min(kpi_score, 0.25)  # Cap at 0.25
    score += kpi_score
    
    # Factor 4: Relationship importance (0-0.15 points)
    relationship_score = 0.0
    for rel in relationships:
        source_col = f"{rel.get('source_table', '')}.{rel.get('source_column', '')}"
        target_col = f"{rel.get('target_table', '')}.{rel.get('target_column', '')}"
        
        if column_key == source_col or column_key == target_col:
            match_rate = rel.get("match_rate", 0.0)
            confidence = rel.get("confidence", "low")
            conf_multiplier = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(confidence, 0.2)
            relationship_score += match_rate * conf_multiplier * 0.05
    relationship_score = min(relationship_score, 0.15)  # Cap at 0.15
    score += relationship_score
    
    # Factor 5: Numeric type bonus (0-0.1 points)
    numeric_type = profile.get("numericType")
    if numeric_type == "measure":
        score += 0.1  # Measures are more important for trends
    elif numeric_type == "dimension":
        score += 0.05  # Dimensions are useful for relationships
    
    return min(score, 1.0)  # Cap at 1.0


def get_top_priority_columns(
    profiles: Dict[str, Any],
    insights: List[Dict[str, Any]],
    kpi_suggestions: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[tuple[str, float]]:
    """Get top priority columns sorted by score.
    
    Args:
        profiles: Dictionary mapping column keys to profiles
        insights: List of insight dictionaries
        kpi_suggestions: List of KPI suggestion dictionaries
        relationships: List of relationship dictionaries
        top_k: Number of top columns to return
    
    Returns:
        List of (column_key, score) tuples sorted by score descending
    """
    scores = []
    
    for column_key, profile in profiles.items():
        score = score_column_priority(
            column_key, profile, insights, kpi_suggestions, relationships
        )
        scores.append((column_key, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


def filter_relationships_by_priority(
    relationships: List[Dict[str, Any]],
    priority_columns: List[str],
) -> List[Dict[str, Any]]:
    """Filter relationships to only include high-priority columns.
    
    Args:
        relationships: List of relationship dictionaries
        priority_columns: List of column keys (e.g., ["table1.col1", "table2.col2"])
    
    Returns:
        Filtered list of relationships
    """
    filtered = []
    
    for rel in relationships:
        source_table = rel.get("source_table", "")
        source_column = rel.get("source_column", "")
        target_table = rel.get("target_table", "")
        target_column = rel.get("target_column", "")
        
        source_key = f"{source_table}.{source_column}"
        target_key = f"{target_table}.{target_column}"
        
        # Include if either source or target is in priority columns
        if source_key in priority_columns or target_key in priority_columns:
            filtered.append(rel)
    
    return filtered

