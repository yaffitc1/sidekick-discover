"""Agent orchestrator for multi-table, multi-source data discovery.

Processes all tables, detects relationships, validates data, suggests KPIs,
and organizes actionable tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from discovery.connectors.snowflake import SnowflakeConnector
from discovery.pipeline.profiler import profile_columns, profile_table
from discovery.pipeline.sampler import advanced_sample
from discovery.analytics.tests import correlation_matrix
from discovery.analytics.relationships import (
    detect_relationships_within_source,
    detect_cross_source_relationships,
    generate_relationship_graph,
)
from discovery.analytics.validation import run_full_validation
from discovery.insights.ranker import generate_insights
from discovery.insights.kpi_suggester import generate_kpi_suggestions
from discovery.insights.task_organizer import organize_tasks
from discovery.render.dashboard import render_dashboard


def _process_table(
    connector,
    table: str,
    source_name: str,
    output_dir: Path,
    sample_rows: int = 100000,
) -> Dict[str, Any]:
    """Process a single table and return all artifacts.
    
    Returns:
        Dictionary with profiles, tests, insights, validation, KPIs
    """
    # Sample data
    df = advanced_sample(connector, table=table, sample_size=sample_rows, corner_case_ratio=0.05)
    
    # Profile
    table_profile = profile_table(df)
    col_profiles = profile_columns(df)
    
    # Tests
    tests = correlation_matrix(df)
    
    # Insights
    insights = generate_insights(col_profiles, tests)
    
    # Validation
    validation = run_full_validation(df)
    
    # KPI suggestions
    kpi_suggestions = generate_kpi_suggestions(df, table_name=table)
    
    # Persist table-specific artifacts
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / f"{table}_table.json").write_text(
        json.dumps(table_profile, indent=2), encoding="utf-8"
    )
    (profiles_dir / f"{table}_columns.json").write_text(
        json.dumps(col_profiles, indent=2), encoding="utf-8"
    )
    
    return {
        "table": table,
        "table_profile": table_profile,
        "col_profiles": col_profiles,
        "tests": tests,
        "insights": insights,
        "validation": validation,
        "kpi_suggestions": kpi_suggestions,
        "sample_df": df,
    }


def _run_pipeline_single_source(
    connector,
    source_name: str,
    output_dir: Path,
    sample_rows: int = 100000,
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the exploration pipeline for a single source.
    
    Processes all tables in the source.
    """
    tables = list(connector.list_tables())
    
    if not tables:
        raise ValueError(f"No tables found for source '{source_name}'")
    
    # Process each table
    table_results = {}
    all_profiles = {}
    all_samples = {}
    all_insights = []
    all_validation_checks = []
    all_kpi_suggestions = []
    
    for table in tables:
        result = _process_table(connector, table, source_name, output_dir, sample_rows)
        table_results[table] = result
        
        all_profiles[table] = result["col_profiles"]
        all_samples[table] = result["sample_df"]
        all_insights.extend(result["insights"])
        all_validation_checks.extend(result["validation"]["checks"])
        
        # Collect KPI suggestions
        for metric in result["kpi_suggestions"].get("simple_metrics", []):
            for suggestion in metric.get("suggestions", []):
                all_kpi_suggestions.append({
                    "table": table,
                    "column": metric["column"],
                    **suggestion,
                })
        # Add dimensions (now includes value-specific KPIs)
        all_kpi_suggestions.extend(result["kpi_suggestions"].get("dimensions", []))
        all_kpi_suggestions.extend(result["kpi_suggestions"].get("composite_kpis", []))
        all_kpi_suggestions.extend(result["kpi_suggestions"].get("trend_kpis", []))
    
    # Detect relationships within source
    relationships = detect_relationships_within_source(all_profiles, all_samples)
    
    # Aggregate results
    aggregated_profiles = {}
    for table, profiles in all_profiles.items():
        aggregated_profiles.update({f"{table}.{k}": v for k, v in profiles.items()})
    
    # Aggregate tests (simple merge for now)
    aggregated_tests = {"pearson": {}, "spearman": {}}
    for table, result in table_results.items():
        tests = result.get("tests", {})
        if "pearson" in tests:
            # Prefix table name to avoid conflicts
            for col, values in tests["pearson"].items():
                aggregated_tests["pearson"][f"{table}.{col}"] = values
    
    return {
        "source_name": source_name,
        "tables": tables,
        "table_results": table_results,
        "profiles": aggregated_profiles,
        "tests": aggregated_tests,
        "insights": all_insights,
        "relationships": relationships,
        "validation_checks": all_validation_checks,
        "kpi_suggestions": all_kpi_suggestions,
        "all_samples": all_samples,  # Include samples for trend analysis
    }


def _run_pipeline_multi_source(
    sources: List[Dict[str, Any]],
    output_dir: Path,
    sample_rows: int = 100000,
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the exploration pipeline across multiple sources.
    
    Processes all tables in all sources and detects cross-source relationships.
    """
    source_results = {}
    all_relationships = []
    all_insights = []
    all_validation_checks = []
    all_kpi_suggestions = []
    
    # Process each source
    for source_config in sources:
        source_name = source_config["name"]
        connector = SnowflakeConnector(
            database=source_config["database"],
            schema=source_config["schema"],
            tables=source_config.get("tables"),
        )
        
        try:
            source_output_dir = output_dir / "sources" / source_name
            result = _run_pipeline_single_source(
                connector, source_name, source_output_dir, sample_rows, goal
            )
            source_results[source_name] = result
            
            all_insights.extend(result["insights"])
            all_validation_checks.extend(result["validation_checks"])
            all_kpi_suggestions.extend(result["kpi_suggestions"])
            all_relationships.extend(result["relationships"])
        finally:
            if connector._conn:
                connector._conn.close()
    
    # Detect cross-source relationships
    source_names = list(source_results.keys())
    for i, source_a_name in enumerate(source_names):
        for source_b_name in source_names[i + 1:]:
            source_a = source_results[source_a_name]
            source_b = source_results[source_b_name]
            
            cross_relationships = detect_cross_source_relationships(
                source_a["profiles"],
                {t: source_a["table_results"][t]["sample_df"] for t in source_a["tables"]},
                source_a_name,
                source_b["profiles"],
                {t: source_b["table_results"][t]["sample_df"] for t in source_b["tables"]},
                source_b_name,
            )
            all_relationships.extend(cross_relationships)
    
    # Organize tasks
    organized_tasks = organize_tasks(
        insights=all_insights,
        relationships=all_relationships,
        validation_checks=all_validation_checks,
        kpi_suggestions=all_kpi_suggestions[:20],  # Limit KPI tasks
    )
    
    # Generate relationship graph
    relationship_graph = generate_relationship_graph(all_relationships)
    
    # Aggregate profiles and tests
    aggregated_profiles = {}
    aggregated_tests = {"pearson": {}, "spearman": {}}
    
    for source_name, source_result in source_results.items():
        aggregated_profiles.update(source_result["profiles"])
        # Merge tests
        for col, values in source_result["tests"].get("pearson", {}).items():
            aggregated_tests["pearson"][col] = values
    
    return {
        "sources": source_names,
        "source_results": source_results,
        "profiles": aggregated_profiles,
        "tests": aggregated_tests,
        "insights": all_insights,
        "relationships": all_relationships,
        "relationship_graph": relationship_graph,
        "validation_checks": all_validation_checks,
        "kpi_suggestions": all_kpi_suggestions,
        "tasks": organized_tasks,
    }


def _run_pipeline(
    connector,
    source_name: str,
    output_dir: str,
    sample_rows: int = 100000,
    goal: Optional[str] = None,
) -> None:
    """Execute the exploration pipeline end-to-end.
    
    Steps: load → sample → profile → correlations → insights → persist → dashboard.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all tables
    result = _run_pipeline_single_source(connector, source_name, out_dir, sample_rows, goal)
    
    # Organize tasks
    organized_tasks = organize_tasks(
        insights=result["insights"],
        relationships=result["relationships"],
        validation_checks=result["validation_checks"],
        kpi_suggestions=result["kpi_suggestions"][:20],
    )
    
    # Generate relationship graph
    relationship_graph = generate_relationship_graph(result["relationships"])
    
    # Persist aggregated artifacts
    (out_dir / "relationships.json").write_text(
        json.dumps(result["relationships"], indent=2), encoding="utf-8"
    )
    (out_dir / "relationship_graph.json").write_text(
        json.dumps(relationship_graph, indent=2), encoding="utf-8"
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps({"checks": result["validation_checks"]}, indent=2), encoding="utf-8"
    )
    (out_dir / "kpi_suggestions.json").write_text(
        json.dumps({"suggestions": result["kpi_suggestions"]}, indent=2), encoding="utf-8"
    )
    (out_dir / "actionable_tasks.json").write_text(
        json.dumps(organized_tasks, indent=2), encoding="utf-8"
    )
    (out_dir / "tests.json").write_text(
        json.dumps(result["tests"], indent=2), encoding="utf-8"
    )
    (out_dir / "insights.md").write_text(
        "\n".join([f"- {i['title']} (score {i.get('score', 0)})" for i in result["insights"]]),
        encoding="utf-8"
    )
    
    # Dashboard (will be enhanced in next step)
    render_dashboard(
        source_name,
        result["profiles"],
        result["tests"],
        result["insights"],
        str(out_dir / "dashboard.html"),
        relationships=result["relationships"],
        relationship_graph=relationship_graph,
        validation_checks=result["validation_checks"],
        kpi_suggestions=result["kpi_suggestions"][:20],
        tasks=organized_tasks,
        overview_stats={
            "total_tables": len(result["tables"]),
            "total_columns": len(result["profiles"]),
            "total_issues": len(result["validation_checks"]) + len(result["insights"]),
        },
        table_samples=result.get("all_samples", {}),
    )


def run_snowflake(
    database: str,
    schema: str,
    source_name: str,
    output_dir: str,
    tables: Optional[List[str]] = None,
    sample_rows: int = 100000,
    goal: Optional[str] = None,
) -> None:
    """Execute the Snowflake exploration pipeline end-to-end."""
    connector = SnowflakeConnector(database=database, schema=schema, tables=tables)
    try:
        _run_pipeline(connector, source_name, output_dir, sample_rows, goal)
    finally:
        if connector._conn:
            connector._conn.close()


def run_multi_source(
    sources: List[Dict[str, Any]],
    output_dir: str,
    sample_rows: int = 100000,
    goal: Optional[str] = None,
) -> None:
    """Execute the exploration pipeline across multiple sources."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result = _run_pipeline_multi_source(sources, out_dir, sample_rows, goal)
    
    # Persist cross-source artifacts
    (out_dir / "relationships.json").write_text(
        json.dumps(result["relationships"], indent=2), encoding="utf-8"
    )
    (out_dir / "relationship_graph.json").write_text(
        json.dumps(result["relationship_graph"], indent=2), encoding="utf-8"
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps({"checks": result["validation_checks"]}, indent=2), encoding="utf-8"
    )
    (out_dir / "kpi_suggestions.json").write_text(
        json.dumps({"suggestions": result["kpi_suggestions"]}, indent=2), encoding="utf-8"
    )
    (out_dir / "actionable_tasks.json").write_text(
        json.dumps(result["tasks"], indent=2), encoding="utf-8"
    )
    (out_dir / "insights.md").write_text(
        "\n".join([f"- {i['title']} (score {i.get('score', 0)})" for i in result["insights"]]),
        encoding="utf-8"
    )
    
    # Dashboard (will be enhanced in next step)
    total_tables = sum(len(sr["tables"]) for sr in result["source_results"].values())
    total_columns = len(result["profiles"])
    total_issues = len(result["validation_checks"]) + len(result["insights"])
    
    # Aggregate table samples from all sources
    all_table_samples = {}
    for source_name, source_result in result["source_results"].items():
        all_table_samples.update(source_result.get("all_samples", {}))
    
    render_dashboard(
        "Multi-Source Analysis",
        result["profiles"],
        result["tests"],
        result["insights"],
        str(out_dir / "dashboard.html"),
        relationships=result["relationships"],
        relationship_graph=result["relationship_graph"],
        validation_checks=result["validation_checks"],
        kpi_suggestions=result["kpi_suggestions"][:20],
        tasks=result["tasks"],
        overview_stats={
            "total_tables": total_tables,
            "total_columns": total_columns,
            "total_issues": total_issues,
        },
        table_samples=all_table_samples,
    )
