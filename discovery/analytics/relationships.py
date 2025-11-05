"""Relationship detection between tables and columns.

Detects foreign key candidates, join opportunities, and referential integrity
across tables and databases.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


def detect_foreign_key_patterns(column_name: str) -> bool:
    """Detect if column name suggests a foreign key relationship.
    
    Patterns: id, _id, code, _code, key, _key suffixes
    """
    name_lower = column_name.lower()
    patterns = ["_id", "_code", "_key", "_pk", "_fk"]
    return any(name_lower.endswith(p) for p in patterns) or name_lower in ["id", "code", "key"]


def analyze_value_overlap(
    source_series: pd.Series,
    target_series: pd.Series,
    sample_limit: int = 10000,
) -> Dict[str, Any]:
    """Analyze value overlap between two columns.
    
    Returns:
        Dictionary with match_rate, overlap_count, and confidence
    """
    # Sample if too large
    if len(source_series) > sample_limit:
        source_series = source_series.sample(n=sample_limit, random_state=42)
    if len(target_series) > sample_limit:
        target_series = target_series.sample(n=sample_limit, random_state=42)
    
    source_values = set(source_series.dropna().astype(str))
    target_values = set(target_series.dropna().astype(str))
    
    if not source_values or not target_values:
        return {
            "match_rate": 0.0,
            "overlap_count": 0,
            "confidence": "low",
        }
    
    overlap = source_values & target_values
    match_rate = len(overlap) / len(source_values) if source_values else 0.0
    
    # Determine confidence
    if match_rate >= 0.95:
        confidence = "high"
    elif match_rate >= 0.7:
        confidence = "medium"
    elif match_rate >= 0.3:
        confidence = "low"
    else:
        confidence = "very_low"
    
    return {
        "match_rate": match_rate,
        "overlap_count": len(overlap),
        "confidence": confidence,
        "source_unique": len(source_values),
        "target_unique": len(target_values),
    }


def detect_relationships_within_source(
    table_profiles: Dict[str, Dict[str, Any]],
    table_samples: Dict[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    """Detect relationships between tables within a single source.
    
    Args:
        table_profiles: Dictionary mapping table names to column profiles
        table_samples: Dictionary mapping table names to sample DataFrames
    
    Returns:
        List of relationship dictionaries
    """
    relationships = []
    table_names = list(table_profiles.keys())
    
    for i, table_a in enumerate(table_names):
        for table_b in table_names[i + 1:]:
            profile_a = table_profiles[table_a]
            profile_b = table_profiles[table_b]
            
            if table_a not in table_samples or table_b not in table_samples:
                continue
            
            df_a = table_samples[table_a]
            df_b = table_samples[table_b]
            
            # Check each column in table A against each column in table B
            for col_a in profile_a.keys():
                if col_a not in df_a.columns:
                    continue
                
                # Check if column name suggests FK
                is_fk_candidate = detect_foreign_key_patterns(col_a)
                
                for col_b in profile_b.keys():
                    if col_b not in df_b.columns:
                        continue
                    
                    # Check data type compatibility
                    dtype_a = str(df_a[col_a].dtype)
                    dtype_b = str(df_b[col_b].dtype)
                    
                    # Skip if types are incompatible
                    if not _are_types_compatible(dtype_a, dtype_b):
                        continue
                    
                    # Analyze value overlap
                    overlap_analysis = analyze_value_overlap(df_a[col_a], df_b[col_b])
                    
                    # If high match rate or FK pattern, record relationship
                    if overlap_analysis["match_rate"] >= 0.3 or is_fk_candidate:
                        relationship = {
                            "source_table": table_a,
                            "source_column": col_a,
                            "target_table": table_b,
                            "target_column": col_b,
                            "match_rate": overlap_analysis["match_rate"],
                            "confidence": overlap_analysis["confidence"],
                            "type": "foreign_key" if overlap_analysis["match_rate"] >= 0.95 else "foreign_key_candidate",
                            "overlap_count": overlap_analysis["overlap_count"],
                        }
                        relationships.append(relationship)
    
    return relationships


def detect_cross_source_relationships(
    source_a_profiles: Dict[str, Dict[str, Any]],
    source_a_samples: Dict[str, pd.DataFrame],
    source_a_name: str,
    source_b_profiles: Dict[str, Dict[str, Any]],
    source_b_samples: Dict[str, pd.DataFrame],
    source_b_name: str,
) -> List[Dict[str, Any]]:
    """Detect relationships between tables across different sources/databases.
    
    Args:
        source_a_profiles: Profiles for source A tables
        source_a_samples: Samples for source A tables
        source_a_name: Name/identifier for source A
        source_b_profiles: Profiles for source B tables
        source_b_samples: Samples for source B tables
        source_b_name: Name/identifier for source B
    
    Returns:
        List of cross-source relationship dictionaries
    """
    relationships = []
    
    for table_a, profile_a in source_a_profiles.items():
        if table_a not in source_a_samples:
            continue
        
        df_a = source_a_samples[table_a]
        
        for table_b, profile_b in source_b_profiles.items():
            if table_b not in source_b_samples:
                continue
            
            df_b = source_b_samples[table_b]
            
            # Check each column combination
            for col_a in profile_a.keys():
                if col_a not in df_a.columns:
                    continue
                
                # Prioritize FK pattern columns
                is_fk_candidate = detect_foreign_key_patterns(col_a)
                
                for col_b in profile_b.keys():
                    if col_b not in df_b.columns:
                        continue
                    
                    # Check type compatibility
                    dtype_a = str(df_a[col_a].dtype)
                    dtype_b = str(df_b[col_b].dtype)
                    
                    if not _are_types_compatible(dtype_a, dtype_b):
                        continue
                    
                    # Analyze overlap
                    overlap_analysis = analyze_value_overlap(df_a[col_a], df_b[col_b])
                    
                    # Lower threshold for cross-database (may be different systems)
                    if overlap_analysis["match_rate"] >= 0.2 or is_fk_candidate:
                        relationship = {
                            "source_table": f"{source_a_name}.{table_a}",
                            "source_column": col_a,
                            "target_table": f"{source_b_name}.{table_b}",
                            "target_column": col_b,
                            "match_rate": overlap_analysis["match_rate"],
                            "confidence": overlap_analysis["confidence"],
                            "type": "foreign_key_candidate" if overlap_analysis["match_rate"] >= 0.7 else "possible_relationship",
                            "overlap_count": overlap_analysis["overlap_count"],
                            "cross_database": True,
                        }
                        relationships.append(relationship)
    
    return relationships


def _are_types_compatible(dtype_a: str, dtype_b: str) -> bool:
    """Check if two data types are compatible for joining."""
    # Normalize types
    dtype_a = dtype_a.lower()
    dtype_b = dtype_b.lower()
    
    # Exact match
    if dtype_a == dtype_b:
        return True
    
    # Numeric types are compatible
    numeric_types = ["int", "float", "bigint", "decimal", "number", "numeric"]
    if any(t in dtype_a for t in numeric_types) and any(t in dtype_b for t in numeric_types):
        return True
    
    # String types are compatible
    string_types = ["object", "string", "varchar", "text", "char"]
    if any(t in dtype_a for t in string_types) and any(t in dtype_b for t in string_types):
        return True
    
    # Date types are compatible
    date_types = ["datetime", "date", "timestamp"]
    if any(t in dtype_a for t in date_types) and any(t in dtype_b for t in date_types):
        return True
    
    return False


def generate_relationship_graph(relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate graph structure for visualization.
    
    Returns:
        Dictionary with nodes and edges for network graph
    """
    nodes = set()
    edges = []
    
    for rel in relationships:
        source_table = rel["source_table"]
        target_table = rel["target_table"]
        
        nodes.add(source_table)
        nodes.add(target_table)
        
        edges.append({
            "source": source_table,
            "target": target_table,
            "source_column": rel["source_column"],
            "target_column": rel["target_column"],
            "match_rate": rel["match_rate"],
            "confidence": rel["confidence"],
            "type": rel["type"],
        })
    
    return {
        "nodes": [{"id": node, "label": node} for node in sorted(nodes)],
        "edges": edges,
    }


def detect_join_opportunities(
    relationships: List[Dict[str, Any]],
    min_match_rate: float = 0.7,
) -> List[Dict[str, Any]]:
    """Extract high-confidence join opportunities from relationships.
    
    Args:
        relationships: List of relationship dictionaries
        min_match_rate: Minimum match rate to consider a join opportunity
    
    Returns:
        List of join opportunity dictionaries with SQL suggestions
    """
    join_opportunities = []
    
    for rel in relationships:
        if rel["match_rate"] >= min_match_rate:
            join_opportunity = {
                "source_table": rel["source_table"],
                "target_table": rel["target_table"],
                "join_column_source": rel["source_column"],
                "join_column_target": rel["target_column"],
                "match_rate": rel["match_rate"],
                "confidence": rel["confidence"],
                "suggested_sql": (
                    f"SELECT * FROM {rel['source_table']} s "
                    f"JOIN {rel['target_table']} t "
                    f"ON s.{rel['source_column']} = t.{rel['target_column']}"
                ),
            }
            join_opportunities.append(join_opportunity)
    
    return join_opportunities





