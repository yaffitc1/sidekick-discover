from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import plotly.graph_objs as go
from plotly.offline import plot as plot_offline
import graphviz
import pandas as pd

from discovery.analytics.column_scoring import (
    get_top_priority_columns,
    filter_relationships_by_priority,
)
from discovery.analytics.time_series import prepare_trend_data
from discovery.analytics.relationships import detect_foreign_key_patterns


def _table_overview(profiles: Dict[str, Any]) -> go.Figure:
    """Bar chart of null percentage per column for quick data hygiene view."""
    cols = list(profiles.keys())
    nulls = [round(float(profiles[c].get("nullPct", 0.0)) * 100, 2) for c in cols]
    fig = go.Figure(data=[go.Bar(x=cols, y=nulls, marker_color="#2e7d32")])
    fig.update_layout(
        title="Null percentage by column",
        xaxis_title="Table.Column",
        yaxis_title="% nulls",
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def _corr_heatmap(tests: Dict[str, Any]) -> go.Figure | None:
    pearson = tests.get("pearson")
    if not pearson:
        return None
    cols = list(pearson.keys())
    if not cols:
        return None
    z = [[float(pearson.get(r, {}).get(c, 0.0)) for c in cols] for r in cols]
    fig = go.Figure(
        data=go.Heatmap(z=z, x=cols, y=cols, colorscale="RdBu", zmin=-1, zmax=1)
    )
    fig.update_layout(
        title="Correlation heatmap (Pearson)",
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def _relationship_graph(
    relationship_graph: Optional[Dict[str, Any]],
    priority_columns: Optional[List[str]] = None,
) -> str:
    """Generate HTML for relationship graph visualization.
    
    Shows relationships filtered to high-priority columns only.
    """
    if not relationship_graph or not relationship_graph.get("edges"):
        return '<p style="color: var(--muted);">No relationships detected.</p>'

    nodes = relationship_graph.get("nodes", [])
    edges = relationship_graph.get("edges", [])

    if not edges:
        return '<p style="color: var(--muted);">No relationships detected.</p>'

    # Filter edges to only include priority columns if provided
    if priority_columns:
        priority_set = set(priority_columns)
        filtered_edges = []
        filtered_nodes = set()
        
        for edge in edges:
            source_table = edge.get("source", "")
            target_table = edge.get("target", "")
            source_col_name = edge.get("source_column", "")
            target_col_name = edge.get("target_column", "")
            
            # Build column keys in format "table.column"
            source_col = f"{source_table}.{source_col_name}" if source_table else source_col_name
            target_col = f"{target_table}.{target_col_name}" if target_table else target_col_name
            
            # Also check without table prefix for flexibility
            source_col_short = source_col_name
            target_col_short = target_col_name
            
            # Include if either source or target matches priority columns
            if (source_col in priority_set or target_col in priority_set or
                source_col_short in priority_set or target_col_short in priority_set):
                filtered_edges.append(edge)
                filtered_nodes.add(source_table)
                filtered_nodes.add(target_table)
        
        edges = filtered_edges
        nodes = [n for n in nodes if n["id"] in filtered_nodes]
        
        if not edges:
            return '<p style="color: var(--muted);">No relationships found among high-priority columns.</p>'

    dot = graphviz.Digraph('ERD', graph_attr={'rankdir': 'LR', 'splines': 'ortho'})

    for node in nodes:
        dot.node(node["id"], label=node["label"], shape='box')

    for edge in edges:
        dot.edge(edge['source'], edge['target'], label=f"{edge['source_column']} -> {edge['target_column']}")

    # Render SVG and return as string
    svg_code = dot.pipe(format='svg').decode('utf-8')

    return f'<div style="text-align: center;">{svg_code}</div>'


def _validation_summary(validation_checks: Optional[List[Dict[str, Any]]]) -> str:
    """Generate HTML for validation summary cards."""
    if not validation_checks:
        return '<p style="color: var(--muted);">No validation issues found.</p>'

    # Count by severity
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for check in validation_checks:
        severity = check.get("severity", "low")
        if severity in by_severity:
            by_severity[severity] += 1

    html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px;">'

    for severity, count in by_severity.items():
        if count > 0:
            color = {
                "critical": "#d32f2f",
                "high": "#f57c00",
                "medium": "#fbc02d",
                "low": "#388e3c"
            }.get(severity, "#757575")

            html += f'''
            <div style="background: {color}; color: white; padding: 16px; border-radius: 8px; text-align: center;">
              <div style="font-size: 24px; font-weight: bold;">{count}</div>
              <div style="font-size: 12px; text-transform: uppercase;">{severity}</div>
            </div>
            '''

    html += '</div>'

    # List top issues
    html += '<div style="max-height: 300px; overflow-y: auto;">'
    for check in validation_checks[:10]:
        severity = check.get("severity", "low")
        message = check.get("message", "Validation issue")
        html += f'''
        <div style="padding: 8px; margin-bottom: 8px; border-left: 3px solid {severity == 'high' and '#f57c00' or '#388e3c'}; background: var(--bg); border-radius: 4px;">
          {message}
        </div>
        '''

    if len(validation_checks) > 10:
        html += f'<p style="font-size: 12px; color: var(--muted);">... and {len(validation_checks) - 10} more issues</p>'

    html += '</div>'
    return html


def _kpi_suggestions(kpi_suggestions: Optional[List[Dict[str, Any]]]) -> str:
    """Generate HTML for KPI suggestion cards."""
    if not kpi_suggestions:
        return '<p style="color: var(--muted);">No KPI suggestions available.</p>'

    html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px;">'

    for kpi in kpi_suggestions[:12]:  # Limit to top 12
        kpi_type = kpi.get("kpi_type", "metric")
        description = kpi.get("description", "Suggested KPI")
        formula = kpi.get("formula", "N/A")
        relevance = kpi.get("relevance_score", 0)

        html += f'''
        <div style="padding: 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--card);">
          <div style="font-weight: 600; margin-bottom: 4px;">{description}</div>
          <div style="font-size: 12px; color: var(--muted); margin-bottom: 8px;">Type: {kpi_type}</div>
          <div style="font-size: 11px; font-family: monospace; background: var(--bg); padding: 6px; border-radius: 4px; margin-bottom: 8px;">{formula}</div>
          <div style="font-size: 11px; color: var(--brand);">Relevance: {relevance:.0%}</div>
        </div>
        '''

    if len(kpi_suggestions) > 12:
        html += f'<div style="grid-column: 1 / -1; padding: 12px; text-align: center; color: var(--muted);">... and {len(kpi_suggestions) - 12} more suggestions</div>'

    html += '</div>'
    return html


def _failed_trends(failed_trends: List[Dict[str, Any]]) -> str:
    """Generate HTML for failed trends with reasons."""
    if not failed_trends:
        return ""
    
    html = '<div style="margin-top: 24px;"><h3>Trends That Could Not Be Generated</h3>'
    html += '<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);">'
    html += '<table style="width: 100%; border-collapse: collapse;">'
    html += '<thead><tr style="border-bottom: 2px solid var(--border);">'
    html += '<th style="text-align: left; padding: 8px;">Column</th>'
    html += '<th style="text-align: left; padding: 8px;">Reason</th>'
    html += '</tr></thead><tbody>'
    
    for failed in failed_trends:
        column_key = failed.get("column_key", "Unknown")
        reason = failed.get("reason", "Unknown reason")
        html += f'<tr style="border-bottom: 1px solid var(--border);">'
        html += f'<td style="padding: 8px; font-family: monospace; font-size: 13px;">{column_key}</td>'
        html += f'<td style="padding: 8px; color: var(--muted);">{reason}</td>'
        html += '</tr>'
    
    html += '</tbody></table></div></div>'
    return html


def _trend_charts(trend_data: List[Dict[str, Any]]) -> str:
    """Generate HTML for time-series trend charts.
    
    One trend per row with trend name in header.
    """
    if not trend_data:
        return '<p style="color: var(--muted);">No trend data available. Ensure tables have datetime columns.</p>'
    
    html = '<div style="display: flex; flex-direction: column; gap: 24px;">'
    
    for trend in trend_data:
        column_key = trend.get("column_key", "Unknown")
        kpi = trend.get("kpi", {})
        data = trend.get("data", [])
        time_col = trend.get("time_column", "")
        metric_col = trend.get("metric_column", "")
        granularity = trend.get("granularity", "day")
        aggregation = trend.get("aggregation", "sum")
        numeric_type = trend.get("numeric_type", "")
        
        if not data:
            html += f'<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);"><h3 style="margin-top: 0;">{column_key}</h3><p style="color: var(--muted);">No data available for this trend.</p></div>'
            continue
        
        # Create DataFrame from data
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            html += f'<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);"><h3 style="margin-top: 0;">{column_key}</h3><p style="color: var(--muted);">Error creating DataFrame: {str(e)}</p></div>'
            continue
        
        if len(df) == 0:
            html += f'<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);"><h3 style="margin-top: 0;">{column_key}</h3><p style="color: var(--muted);">Empty DataFrame after conversion.</p></div>'
            continue
        
        # Ensure time column is datetime (it might be string after dict conversion)
        if time_col not in df.columns:
            html += f'<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);"><h3 style="margin-top: 0;">{column_key}</h3><p style="color: var(--muted);">Time column "{time_col}" not found in data. Available columns: {list(df.columns)}</p></div>'
            continue
            
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
        # Ensure metric column is numeric
        if metric_col not in df.columns:
            html += f'<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);"><h3 style="margin-top: 0;">{column_key}</h3><p style="color: var(--muted);">Metric column "{metric_col}" not found in data. Available columns: {list(df.columns)}</p></div>'
            continue
            
        if not pd.api.types.is_numeric_dtype(df[metric_col]):
            df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        
        # Drop any rows with invalid data
        df = df.dropna()
        
        if len(df) == 0:
            html += f'<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);"><h3 style="margin-top: 0;">{column_key}</h3><p style="color: var(--muted);">No valid data points after filtering nulls.</p></div>'
            continue
        
        # Create Plotly figure
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[metric_col],
                mode='lines+markers',
                name=metric_col,
                line=dict(color='#2e7d32', width=2),
                marker=dict(size=6),
            ))
            
            type_label = f" ({numeric_type})" if numeric_type else ""
            kpi_description = kpi.get('description', 'Trend') if kpi else 'Trend'
            
            fig.update_layout(
                title=f"{kpi_description}",
                xaxis_title=f"Time ({granularity})",
                yaxis_title=f"{metric_col} ({aggregation})",
                margin=dict(l=40, r=20, t=50, b=40),
                template="plotly_white",
                height=350,
            )
            
            chart_div = plot_offline(fig, include_plotlyjs=False, output_type="div")
            
            # Create section with header
            html += f'''
            <div style="background: var(--card); padding: 20px; border-radius: 8px; border: 1px solid var(--border);">
                <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 18px; color: var(--ink);">{column_key}{type_label}</h3>
                <div>{chart_div}</div>
            </div>
            '''
        except Exception as e:
            html += f'<div style="background: var(--card); padding: 16px; border-radius: 8px; border: 1px solid var(--border);"><h3 style="margin-top: 0;">{column_key}</h3><p style="color: var(--muted);">Error creating chart: {str(e)}</p></div>'
            continue
    
    html += '</div>'
    return html


def _actionable_tasks(tasks: Optional[Dict[str, Any]]) -> str:
    """Generate HTML for actionable tasks organized by category."""
    if not tasks or not tasks.get("tasks"):
        return '<p style="color: var(--muted);">No actionable tasks generated.</p>'

    organized_tasks = tasks.get("tasks", {})
    summary = tasks.get("summary", {})

    html = f'<div style="margin-bottom: 16px;"><p><strong>Total: {summary.get("total_tasks", 0)} tasks</strong></p></div>'

    categories = ["data_quality", "relationships", "kpis", "optimization"]
    category_labels = {
        "data_quality": "Data Quality",
        "relationships": "Relationships",
        "kpis": "KPIs",
        "optimization": "Optimization"
    }

    priority_colors = {
        "critical": "#d32f2f",
        "high": "#f57c00",
        "medium": "#fbc02d",
        "low": "#388e3c"
    }

    for category in categories:
        task_list = organized_tasks.get(category, [])
        if not task_list:
            continue

        html += f'<div style="margin-bottom: 24px;"><h3 style="margin-bottom: 12px;">{category_labels.get(category, category)}</h3>'

        for task in task_list[:10]:  # Limit to top 10 per category
            priority = task.get("priority", "low")
            title = task.get("title", "Untitled task")
            actions = task.get("actions", [])

            html += f'''
            <div style="padding: 12px; margin-bottom: 8px; border-left: 4px solid {priority_colors.get(priority, "#757575")}; background: var(--card); border-radius: 6px;">
              <div style="display: flex; align-items: start; gap: 8px;">
                <input type="checkbox" style="margin-top: 4px;">
                <div style="flex: 1;">
                  <div style="font-weight: 600; margin-bottom: 4px;">{title}</div>
                  <div style="font-size: 12px; color: var(--muted); margin-bottom: 8px;">{task.get("rationale", "")}</div>
                  <div style="font-size: 11px;">
                    <strong>Actions:</strong>
                    <ul style="margin: 4px 0; padding-left: 20px;">
                      {''.join([f"<li>{action}</li>" for action in actions[:3]])}
                    </ul>
                  </div>
                </div>
                <span style="background: {priority_colors.get(priority, "#757575")}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{priority}</span>
              </div>
            </div>
            '''

        if len(task_list) > 10:
            html += f'<p style="font-size: 12px; color: var(--muted);">... and {len(task_list) - 10} more tasks</p>'

        html += '</div>'

    return html


def render_dashboard(
    source_name: str,
    profiles: Dict[str, Any],
    tests: Dict[str, Any],
    insights: List[Dict[str, Any]],
    out_html: str,
    relationships: Optional[List[Dict[str, Any]]] = None,
    relationship_graph: Optional[Dict[str, Any]] = None,
    validation_checks: Optional[List[Dict[str, Any]]] = None,
    kpi_suggestions: Optional[List[Dict[str, Any]]] = None,
    tasks: Optional[Dict[str, Any]] = None,
    overview_stats: Optional[Dict[str, Any]] = None,
    table_samples: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Render an offline HTML dashboard with a green, clean theme.
    
    Enhanced with relationships, validation, KPIs, actionable tasks, and trends sections.
    
    Args:
        table_samples: Dictionary mapping table names to sample DataFrames for trend analysis
    """
    out_path = Path(out_html).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Data for summaries
    
    # Sidebar
    sidebar_html = _generate_sidebar(profiles, source_name)

    # Summaries
    summaries_html = _generate_summaries(
        profiles, source_name, table_samples, tests, kpi_suggestions or []
    )


    bar_fig = _table_overview(profiles)
    # Inline plotly.js for offline viewing
    bar_div = plot_offline(bar_fig, include_plotlyjs=True, output_type="div")

    corr_fig = _corr_heatmap(tests)
    corr_div = (
        plot_offline(corr_fig, include_plotlyjs=False, output_type="div") if corr_fig else ""
    )

    insights_items = "".join(
        [
            f"<li><span class=\"badge\">Score {int(i.get('score', 0))}</span> {i['title']}</li>"
            for i in insights
        ]
    )

    # Calculate dimensions vs measures from profiles
    dimensions_count = 0
    measures_count = 0
    for profile in profiles.values():
        numeric_type = profile.get("numericType")
        if numeric_type == "dimension":
            dimensions_count += 1
        elif numeric_type == "measure":
            measures_count += 1

    # Score columns and get top priority columns
    priority_columns_data = get_top_priority_columns(
        profiles, insights, kpi_suggestions or [], relationships or []
    )
    priority_columns = [col for col, score in priority_columns_data]
    
    # Filter relationships to priority columns
    filtered_relationships = relationships
    if relationships and priority_columns:
        filtered_relationships = filter_relationships_by_priority(relationships, priority_columns)
        # Regenerate relationship graph with filtered relationships
        if filtered_relationships:
            from discovery.analytics.relationships import generate_relationship_graph
            relationship_graph = generate_relationship_graph(filtered_relationships)
    
    # Prepare trend data for high-priority columns (includes all columns with time/date)
    # Note: This is used for per-column pages, not main dashboard
    trend_result = {}
    if table_samples and priority_columns:
        trend_result = prepare_trend_data(
            table_samples, priority_columns, kpi_suggestions or [], profiles, top_kpis_per_column=1
        )
    
    trend_data = trend_result.get("trends", [])
    failed_trends = trend_result.get("failed", [])
    
    # Generate trend HTML for per-column pages (not used in main dashboard)
    trends_html = _trend_charts(trend_data)
    failed_trends_html = _failed_trends(failed_trends)
    
    # Generate new sections
    overview_html = ""
    if overview_stats:
        total_tables = overview_stats.get("total_tables", 0)
        total_columns = overview_stats.get("total_columns", 0)
        total_issues = overview_stats.get("total_issues", 0)
        overview_html = f'''
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
          <div style="background: var(--card); padding: 20px; border-radius: 8px; text-align: center; border: 1px solid var(--border);">
            <div style="font-size: 32px; font-weight: bold; color: var(--brand);">{total_tables}</div>
            <div style="font-size: 14px; color: var(--muted);">Tables</div>
          </div>
          <div style="background: var(--card); padding: 20px; border-radius: 8px; text-align: center; border: 1px solid var(--border);">
            <div style="font-size: 32px; font-weight: bold; color: var(--brand);">{total_columns}</div>
            <div style="font-size: 14px; color: var(--muted);">Columns</div>
          </div>
          <div style="background: var(--card); padding: 20px; border-radius: 8px; text-align: center; border: 1px solid var(--border);">
            <div style="font-size: 32px; font-weight: bold; color: var(--brand);">{total_issues}</div>
            <div style="font-size: 14px; color: var(--muted);">Issues Found</div>
          </div>
        </div>
        '''
    
    # Add dimensions vs measures section if we have numeric columns
    if dimensions_count > 0 or measures_count > 0:
        dimensions_html = f'''
        <div style="margin-top: 24px;">
          <h3 style="margin-bottom: 16px; font-size: 16px; color: var(--ink);">Numeric Column Classification</h3>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
            <div style="background: var(--card); padding: 20px; border-radius: 8px; text-align: center; border: 1px solid var(--border);">
              <div style="font-size: 32px; font-weight: bold; color: #2e7d32;">{dimensions_count}</div>
              <div style="font-size: 14px; color: var(--muted);">Dimensions</div>
              <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">For grouping/filtering</div>
            </div>
            <div style="background: var(--card); padding: 20px; border-radius: 8px; text-align: center; border: 1px solid var(--border);">
              <div style="font-size: 32px; font-weight: bold; color: #1976d2;">{measures_count}</div>
              <div style="font-size: 14px; color: var(--muted);">Measures</div>
              <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">For aggregation</div>
            </div>
          </div>
        </div>
        '''
        overview_html = overview_html + dimensions_html if overview_html else dimensions_html

    relationships_html = _relationship_graph(relationship_graph, priority_columns)
    
    # Add priority column scores display
    if priority_columns_data:
        priority_html = '<div style="margin-bottom: 16px;"><h3 style="font-size: 14px; margin-bottom: 8px;">Top Priority Columns</h3><div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        for col, score in priority_columns_data[:10]:
            priority_html += f'<span style="background: var(--brand-300); color: white; padding: 4px 12px; border-radius: 16px; font-size: 12px;">{col} ({score:.2f})</span>'
        priority_html += '</div></div>'
        relationships_html = priority_html + relationships_html
    
    validation_html = _validation_summary(validation_checks)
    kpi_html = _kpi_suggestions(kpi_suggestions)
    tasks_html = _actionable_tasks(tasks)
    
    # Note: trends_html and failed_trends_html are only used in per-column pages, not main dashboard
    
    # Build section HTML
    overview_section = f'<section id="overview" class="section grid"><div class="card" style="grid-column: 1 / -1;"><div class="head"><h2>Overview</h2></div><div class="body">{overview_html}</div></div></section>' if overview_html else ''
    relationships_section = f'<section id="relationships" class="section grid"><div class="card" style="grid-column: 1 / -1;"><div class="head"><h2>Relationships (High Priority Columns)</h2></div><div class="body">{relationships_html}</div></div></section>' if relationships_html else ''
    validation_section = f'<section id="validation" class="section grid"><div class="card" style="grid-column: 1 / -1;"><div class="head"><h2>Data Validation</h2></div><div class="body">{validation_html}</div></div></section>' if validation_html else ''
    kpi_section = f'<section id="kpis" class="section grid"><div class="card" style="grid-column: 1 / -1;"><div class="head"><h2>KPI Suggestions</h2></div><div class="body">{kpi_html}</div></div></section>' if kpi_html else ''
    tasks_section = f'<section id="tasks" class="section grid"><div class="card" style="grid-column: 1 / -1;"><div class="head"><h2>Actionable Tasks</h2></div><div class="body">{tasks_html}</div></div></section>' if tasks_html else ''

    html = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Discovery — {source_name}</title>
    <style>
      :root {{
        --brand: #2e7d32; /* primary green */
        --brand-600: #1b5e20;
        --brand-300: #66bb6a;
        --ink: #1b1b1b;
        --muted: #5f6a6a;
        --bg: #f7faf7;
        --card: #ffffff;
        --border: #e8ede8;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif;
        color: var(--ink);
        background: var(--bg);
        margin: 0;
        display: flex;
      }}
      .sidebar {{
        width: 280px;
        background: var(--card);
        border-right: 1px solid var(--border);
        padding: 20px;
        height: 100vh;
        overflow-y: auto;
        position: fixed;
      }}
      .main-content {{
        flex: 1;
        padding-left: 280px;
      }}
      .header {{
        background: linear-gradient(135deg, var(--brand) 0%, var(--brand-300) 100%);
        color: #fff;
        padding: 24px 20px 48px;
      }}
      .hero {{ max-width: 1120px; margin: 24px auto 0; padding: 0 20px; }}
      .hero h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
      .hero p {{ margin: 0; opacity: 0.9; }}
      .container {{ max-width: 1120px; margin: -24px auto 40px; padding: 0 20px; }}
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
      @media (min-width: 960px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
      .section {{ margin-top: 24px; }}
      .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.04); }}
      .card .head {{ padding: 16px 16px 0 16px; }}
      .card .head h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
      .card .body {{ padding: 8px 16px 16px 16px; }}
      .list {{ list-style: none; padding: 0; margin: 0; }}
      .list li {{ padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }}
      .badge {{ background: var(--brand); color: #fff; border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 600; }}
      .footer {{ text-align: center; color: var(--muted); padding: 24px; font-size: 13px; }}
      .nav {{
        display: flex; align-items: center; justify-content: space-between;
        max-width: 1120px; margin: 0 auto;
        flex-wrap: wrap;
        gap: 12px;
      }}
      .brand {{ display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: 0.3px; }}
      .brand span {{ opacity: 0.9; font-weight: 600; }}
      .nav a {{ color: #e8ffe8; text-decoration: none; margin-left: 16px; font-weight: 500; }}
      .hero {{ max-width: 1120px; margin: 24px auto 0; padding: 0 20px; }}
      .hero h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
      .hero p {{ margin: 0; opacity: 0.9; }}
      .container {{ max-width: 1120px; margin: -24px auto 40px; padding: 0 20px; }}
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
      @media (min-width: 960px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
      .section {{ margin-top: 24px; }}
      .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.04); }}
      .card .head {{ padding: 16px 16px 0 16px; }}
      .card .head h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
      .card .body {{ padding: 8px 16px 16px 16px; }}
      .list {{ list-style: none; padding: 0; margin: 0; }}
      .list li {{ padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }}
      .badge {{ background: var(--brand); color: #fff; border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 600; }}
      .footer {{ text-align: center; color: var(--muted); padding: 24px; font-size: 13px; }}
      .anchors {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
      .anchors a {{ color: #fff; opacity: 0.95; font-size: 13px; white-space: nowrap; }}
      .anchors a:hover {{ opacity: 1; text-decoration: underline; }}
      .sidebar-link {{ color: var(--ink); text-decoration: none; display: block; padding: 4px 8px; border-radius: 4px; transition: background 0.2s; }}
      .sidebar-link:hover {{ background: var(--bg); }}
      .sidebar-link.highlighted {{ background: #ffeb3b; font-weight: 600; }}
      .sidebar ul {{ list-style: none; padding-left: 16px; margin: 4px 0; }}
      .sidebar li {{ margin: 2px 0; }}
      .search-box {{
        position: relative;
        display: flex;
        align-items: center;
      }}
      .search-box input {{
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 8px 16px 8px 36px;
        color: #fff;
        font-size: 14px;
        width: 200px;
        outline: none;
        transition: all 0.2s;
      }}
      .search-box input::placeholder {{
        color: rgba(255, 255, 255, 0.7);
      }}
      .search-box input:focus {{
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
        width: 250px;
      }}
      .search-box::before {{
        content: '🔍';
        position: absolute;
        left: 12px;
        font-size: 14px;
        pointer-events: none;
      }}
      .search-highlight {{
        background: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
      }}
      .section-hidden {{
        display: none !important;
      }}
    </style>
  </head>
  <body>
    <aside class="sidebar">
        {sidebar_html}
    </aside>
    <div class="main-content">
        <header class="header">
        <div class="nav">
            <div class="brand">
            <svg width="28" height="28" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <circle cx="32" cy="32" r="30" stroke="#e8ede8" stroke-width="2"/>
                <path d="M16 40 C18 28, 28 20, 40 22 C46 23, 50 28, 52 34" stroke="#ffffff" stroke-width="3" fill="none" stroke-linecap="round"/>
                <circle cx="24" cy="28" r="3" fill="#e8ede8"/>
                <circle cx="40" cy="30" r="3" fill="#e8ede8"/>
            </svg>
            Discovery <span>dashboard</span>
            </div>
            <nav class="anchors">
            <div class="search-box">
                <input type="text" id="search-input" placeholder="Search dashboard..." />
            </div>
            <a href="#overview">Overview</a>
            <a href="#insights">Insights</a>
            <a href="#relationships">Relationships</a>
            <a href="#validation">Validation</a>
            <a href="#kpis">KPIs</a>
            <a href="#tasks">Tasks</a>
            <a href="#nulls">Nulls</a>
            </nav>
        </div>
        <div class="hero">
            <h1>Source: {source_name}</h1>
            <p>Automated profiles, statistical tests and ranked insights.</p>
        </div>
        </header>

        <main class="container">
        <div id="dashboard-content">
            {overview_section}
            
            <section id="insights" class="section grid">
                <div class="card" style="grid-column: 1 / -1;">
                <div class="head"><h2>Top insights</h2></div>
                <div class="body">
                    <ul class="list">{insights_items if insights_items else '<li>No insights generated.</li>'}</ul>
                </div>
                </div>
            </section>

            {relationships_section}

            {validation_section}

            {kpi_section}

            {tasks_section}

            <section id="nulls" class="section grid">
                <div class="card" style="grid-column: 1 / -1;">
                <div class="head"><h2>Nulls overview</h2></div>
                <div class="body">{bar_div}</div>
                </div>
            </section>
        </div>
        <div id="summary-content" style="display: none;">
            {summaries_html}
        </div>
        </main>

        <footer class="footer">Generated by Discovery</footer>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const sidebarLinks = document.querySelectorAll('.sidebar-link');
            const dashboardContent = document.getElementById('dashboard-content');
            const summaryContent = document.getElementById('summary-content');
            const summaries = document.querySelectorAll('.summary-view');
            const searchInput = document.getElementById('search-input');

            sidebarLinks.forEach(link => {{
                link.addEventListener('click', function(e) {{
                    e.preventDefault();
                    const targetId = this.dataset.target;

                    if (targetId === 'dashboard') {{
                        dashboardContent.style.display = 'block';
                        summaryContent.style.display = 'none';
                    }} else {{
                        dashboardContent.style.display = 'none';
                        summaryContent.style.display = 'block';

                        summaries.forEach(summary => {{
                            if (summary.id === targetId) {{
                                summary.style.display = 'block';
                            }} else {{
                                summary.style.display = 'none';
                            }}
                        }});
                    }}
                }});
            }});

            // Search functionality for sidebar menu items
            function highlightText(text, searchTerm) {{
                if (!searchTerm) return text;
                const regex = new RegExp(`(${{searchTerm}})`, 'gi');
                return text.replace(regex, '<span class="search-highlight">$1</span>');
            }}

            function performSearch(query) {{
                const searchLower = query.toLowerCase().trim();
                const sidebarLinks = document.querySelectorAll('.sidebar-link');
                let firstMatch = null;

                if (!searchLower) {{
                    // Clear highlights
                    sidebarLinks.forEach(link => {{
                        link.classList.remove('highlighted');
                        const originalText = link.textContent;
                        link.innerHTML = originalText;
                    }});
                    // Also search sections
                    const sections = document.querySelectorAll('.section');
                    sections.forEach(section => {{
                        section.classList.remove('section-hidden');
                        section.querySelectorAll('.search-highlight').forEach(el => {{
                            const parent = el.parentNode;
                            parent.replaceChild(document.createTextNode(el.textContent), el);
                            parent.normalize();
                        }});
                    }});
                    return;
                }}

                // Search sidebar menu items
                sidebarLinks.forEach(link => {{
                    const linkText = link.textContent.toLowerCase();
                    const hasMatch = linkText.includes(searchLower);
                    
                    if (hasMatch) {{
                        link.classList.add('highlighted');
                        // Highlight matching text
                        const originalText = link.textContent;
                        link.innerHTML = highlightText(originalText, query);
                        
                        // Track first match for scrolling
                        if (!firstMatch) {{
                            firstMatch = link;
                        }}
                    }} else {{
                        link.classList.remove('highlighted');
                        const originalText = link.textContent;
                        link.innerHTML = originalText;
                    }}
                }});

                // Scroll to first match in sidebar
                if (firstMatch) {{
                    firstMatch.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}

                // Also search sections
                const sections = document.querySelectorAll('.section');
                sections.forEach(section => {{
                    const text = section.textContent.toLowerCase();
                    const hasMatch = text.includes(searchLower);

                    if (hasMatch) {{
                        section.classList.remove('section-hidden');
                        // Highlight matching text in headings
                        const headings = section.querySelectorAll('h2, h3');
                        headings.forEach(heading => {{
                            const originalText = heading.textContent;
                            heading.innerHTML = highlightText(originalText, query);
                        }});
                    }} else {{
                        section.classList.add('section-hidden');
                    }}
                }});
            }}

            searchInput.addEventListener('input', function(e) {{
                performSearch(e.target.value);
            }});

            searchInput.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    e.target.value = '';
                    performSearch('');
                }}
            }});
        }});
    </script>
  </body>
 </html>
"""
    out_path.write_text(html, encoding="utf-8")


def _generate_sidebar(profiles: Dict[str, Any], source_name: str) -> str:
    """Generate HTML for the sidebar navigation."""
    
    # Simple list for now
    html = "<h3>Navigation</h3>"
    html += f'<ul><li><a href="#" class="sidebar-link" data-target="dashboard">Dashboard</a></li>'
    
    # Reconstruct hierarchy from profiles
    structure = {}
    for key in profiles.keys():
        parts = key.split('.')
        if len(parts) == 2:
            table, column = parts
            if table not in structure:
                structure[table] = []
            structure[table].append(column)

    # Build HTML
    for table, columns in sorted(structure.items()):
        html += f'<li><a href="#" class="sidebar-link" data-target="summary-{table}">{table}</a><ul>'
        for col in sorted(columns):
            html += f'<li><a href="#" class="sidebar-link" data-target="summary-{table}-{col}">{col}</a></li>'
        html += '</ul></li>'

    html += '</ul>'
    return html


def _is_id_or_code_column(column_name: str) -> bool:
    """Detect if column name suggests an ID or code column.
    
    Patterns: id, _id, code, _code, key, _key suffixes
    """
    return detect_foreign_key_patterns(column_name)


def _is_categorical_column(df: pd.DataFrame, column: str) -> bool:
    """Detect if column is categorical (object type with reasonable cardinality)."""
    if column not in df.columns:
        return False
    
    series = df[column]
    # Categorical: object type or category dtype with reasonable cardinality
    if series.dtype == "object" or series.dtype.name == "category":
        nunique = series.nunique()
        total = len(series)
        # Good cardinality for categorical: 2-100 distinct values
        if 2 <= nunique <= 100:
            return True
    return False


def _get_top_bottom_categorical_values(df: pd.DataFrame, column: str, top_n: int = 5) -> List[Tuple[Any, int]]:
    """Get top N and bottom N categorical values by occurrence count.
    
    Returns:
        List of tuples (value, count) sorted by count descending
    """
    if column not in df.columns:
        return []
    
    value_counts = df[column].value_counts(dropna=True)
    
    # Get top N and bottom N
    top_values = value_counts.head(top_n).items()
    bottom_values = value_counts.tail(top_n).items()
    
    # Combine and deduplicate (in case there are fewer than 2*top_n unique values)
    all_values = list(set(list(top_values) + list(bottom_values)))
    
    # Sort by count descending
    all_values.sort(key=lambda x: x[1], reverse=True)
    
    return all_values


def _generate_categorical_trend_charts(
    df: pd.DataFrame,
    column: str,
    time_column: str,
    categorical_values: List[Tuple[Any, int]],
    granularity: str,
) -> str:
    """Generate trend charts for each categorical value.
    
    Shows number of occurrences of each value over time.
    """
    html = ""
    
    from discovery.analytics.time_series import aggregate_kpi_over_time
    
    for value, total_count in categorical_values:
        # Create a binary column: 1 if value matches, 0 otherwise
        df_filtered = df.copy()
        df_filtered['_temp_match'] = (df_filtered[column] == value).astype(int)
        
        # Aggregate count over time
        try:
            trend_df = aggregate_kpi_over_time(
                df_filtered, time_column, '_temp_match', "sum", granularity
            )
            
            if len(trend_df) == 0:
                continue
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df[time_column],
                y=trend_df['_temp_match'],
                mode='lines+markers',
                name=str(value),
                line=dict(color='#2e7d32', width=2),
                marker=dict(size=6),
            ))
            
            # Format value for display
            value_str = str(value) if pd.notna(value) else "NULL"
            formula = f"COUNT({column} = '{value_str}')"
            
            fig.update_layout(
                title=f"Occurrences of '{value_str}' in {column}",
                xaxis_title=f"Time ({granularity}) - Period Column: {time_column}",
                yaxis_title=f"Number of Occurrences",
                margin=dict(l=40, r=20, t=50, b=40),
                template="plotly_white",
                height=300,
            )
            
            chart_div = plot_offline(fig, include_plotlyjs=False, output_type="div")
            
            html += f'''
            <div style="margin-bottom: 24px;">
                <div style="margin-bottom: 12px; padding: 12px; background: var(--bg); border-radius: 6px; font-size: 13px;">
                    <div><strong>Value:</strong> <code>{value_str}</code> (Total occurrences: {total_count})</div>
                    <div style="margin-top: 4px;"><strong>Period Column:</strong> <code>{time_column}</code></div>
                    <div style="margin-top: 4px;"><strong>Formula:</strong> <code>{formula}</code></div>
                </div>
                {chart_div}
            </div>
            '''
        except Exception as e:
            # Skip this value if aggregation fails
            continue
    
    return html


def _generate_column_trend_chart(
    column_key: str,
    table_samples: Optional[Dict[str, pd.DataFrame]],
    kpi_suggestions: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> str:
    """Generate trend chart HTML for a specific column.
    
    Always returns HTML - either a chart or an explanation of why it couldn't be generated.
    Handles:
    - ID/code columns: Uses COUNT DISTINCT
    - Categorical columns: Shows top 5 and bottom 5 values with separate graphs
    - Numeric columns: Uses standard aggregations
    """
    html = '<div style="margin-top: 20px;"><h4>Trend Analysis</h4>'
    
    if not table_samples:
        html += '<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">No table samples available for trend analysis.</p></div>'
        return html
    
    if "." not in column_key:
        html += '<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Invalid column key format (missing table prefix).</p></div>'
        return html
    
    table, column = column_key.split(".", 1)
    
    if table not in table_samples:
        html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Table "{table}" not found in samples.</p></div>'
        return html
    
    df = table_samples[table]
    
    if column not in df.columns:
        html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Column "{column}" not found in table "{table}".</p></div>'
        return html
    
    # Detect time column
    from discovery.analytics.time_series import (
        detect_primary_time_column,
        get_best_kpi_for_column,
        select_optimal_granularity,
        aggregate_kpi_over_time,
    )
    
    time_column = detect_primary_time_column(df)
    if not time_column:
        html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">No datetime/timestamp columns found in table "{table}". Cannot generate trend without a time dimension.</p></div>'
        return html
    
    # Check if categorical column
    if _is_categorical_column(df, column):
        granularity = select_optimal_granularity(df, time_column)
        categorical_values = _get_top_bottom_categorical_values(df, column, top_n=5)
        
        if not categorical_values:
            html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">No categorical values found for column "{column}".</p></div>'
            return html
        
        html += _generate_categorical_trend_charts(df, column, time_column, categorical_values, granularity)
        html += '</div>'
        return html
    
    # Check if ID/code column
    is_id_code = _is_id_or_code_column(column)
    
    # Get best KPI
    best_kpi = get_best_kpi_for_column(column_key, kpi_suggestions, profile)
    
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
            # For boolean columns, create a sum KPI
            best_kpi = {
                "kpi_type": "sum",
                "description": f"Count of {column} (true values)",
                "formula": f"SUM({column})",
                "column": column,
                "relevance_score": 0.7,
            }
        else:
            html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">No KPI suggestions found for column "{column}" and column is not numeric or boolean. Cannot generate trend.</p></div>'
            return html
    
    # Get metric column
    metric_col = best_kpi.get("metric_column") or best_kpi.get("column") or column
    if metric_col not in df.columns:
        metric_col_short = metric_col.split(".")[-1] if "." in metric_col else metric_col
        if metric_col_short in df.columns:
            metric_col = metric_col_short
        else:
            html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Metric column "{metric_col}" not found in table.</p></div>'
            return html
    
    # Skip if metric column is datetime or same as time column
    if pd.api.types.is_datetime64_any_dtype(df[metric_col]):
        html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Metric column "{metric_col}" is datetime type. Cannot aggregate datetime columns as metrics.</p></div>'
        return html
    
    if metric_col == time_column:
        html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Metric column "{metric_col}" is the same as time column "{time_column}". Cannot use the same column for both time and metric.</p></div>'
        return html
    
    # For ID/code columns, use count_distinct (don't need to convert to numeric)
    if is_id_code:
        aggregation = "count_distinct"
        kpi_formula = best_kpi.get("formula", f"COUNT(DISTINCT {metric_col})")
    else:
        # Ensure metric column is numeric
        if not pd.api.types.is_numeric_dtype(df[metric_col]):
            if df[metric_col].dtype == 'bool':
                # Convert boolean to int: true=1, false=0, null=0
                df[metric_col] = df[metric_col].fillna(False).astype(int)
            else:
                # Try to convert to numeric
                try:
                    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
                    if df[metric_col].isna().all():
                        html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Metric column "{metric_col}" cannot be converted to numeric (all values are invalid).</p></div>'
                        return html
                except Exception as e:
                    html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Error converting metric column "{metric_col}" to numeric: {str(e)}</p></div>'
                    return html
        
        # Determine aggregation
        kpi_type = best_kpi.get("kpi_type", "sum")
        aggregation_map = {
            "sum": "sum",
            "average": "avg",
            "count": "count",
            "trend": "sum",
            "ratio": "avg",
            "percentage": "avg",
        }
        # For boolean columns, always use sum
        if df[metric_col].dtype == 'bool' or (hasattr(df[metric_col], 'dtype') and df[metric_col].dtype == 'int64' and df[metric_col].isin([0, 1]).all()):
            aggregation = "sum"
        else:
            aggregation = aggregation_map.get(kpi_type, "sum")
        
        kpi_formula = best_kpi.get("formula", f"{aggregation.upper()}({metric_col})")
    
    # Ensure time column is datetime for filtering
    df_time = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_time[time_column]):
        if df_time[time_column].dtype == 'object':
            df_time[time_column] = pd.to_datetime(df_time[time_column], errors="coerce")
        else:
            df_time[time_column] = pd.to_datetime(df_time[time_column], errors="coerce")
    
    # Aggregate over time - Yearly trend for all data
    try:
        # Use yearly granularity for full trend
        yearly_granularity = "year"
        trend_df_yearly = aggregate_kpi_over_time(df_time, time_column, metric_col, aggregation, yearly_granularity)
        
        # Filter to last 2 years for monthly trend
        if len(df_time) > 0 and pd.api.types.is_datetime64_any_dtype(df_time[time_column]):
            max_date = df_time[time_column].max()
            if pd.notna(max_date):
                two_years_ago = max_date - pd.DateOffset(years=2)
                df_last_2_years = df_time[df_time[time_column] >= two_years_ago]
            else:
                df_last_2_years = pd.DataFrame()
        else:
            df_last_2_years = pd.DataFrame()
        
        # Generate monthly trend for last 2 years
        monthly_granularity = "month"
        trend_df_monthly = pd.DataFrame()
        if len(df_last_2_years) > 0:
            trend_df_monthly = aggregate_kpi_over_time(df_last_2_years, time_column, metric_col, aggregation, monthly_granularity)
        
        # Generate yearly chart
        if len(trend_df_yearly) == 0:
            html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Time aggregation returned empty result. No valid data points after aggregating "{metric_col}" over "{time_column}" with {yearly_granularity} granularity.</p></div>'
            return html
        
        kpi_description = best_kpi.get("description", f"Trend of {metric_col}")
        
        # Yearly trend chart (all data)
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Scatter(
            x=trend_df_yearly[time_column],
            y=trend_df_yearly[metric_col],
            mode='lines+markers',
            name=metric_col,
            line=dict(color='#2e7d32', width=2),
            marker=dict(size=6),
        ))
        
        fig_yearly.update_layout(
            title=f"{kpi_description} - Yearly Trend (All Time)",
            xaxis_title=f"Time ({yearly_granularity}) - Period Column: {time_column}",
            yaxis_title=f"{metric_col} ({aggregation})",
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=300,
        )
        
        chart_div_yearly = plot_offline(fig_yearly, include_plotlyjs=False, output_type="div")
        
        html += f'''
        <div style="margin-bottom: 24px;">
            <div style="margin-bottom: 12px; padding: 12px; background: var(--bg); border-radius: 6px; font-size: 13px;">
                <div><strong>Period Column:</strong> <code>{time_column}</code></div>
                <div style="margin-top: 4px;"><strong>Formula:</strong> <code>{kpi_formula}</code></div>
                <div style="margin-top: 4px;"><strong>Granularity:</strong> Yearly (All Time)</div>
            </div>
            {chart_div_yearly}
        </div>
        '''
        
        # Monthly trend chart (last 2 years)
        if len(trend_df_monthly) > 0:
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Scatter(
                x=trend_df_monthly[time_column],
                y=trend_df_monthly[metric_col],
                mode='lines+markers',
                name=metric_col,
                line=dict(color='#1976d2', width=2),
                marker=dict(size=6),
            ))
            
            fig_monthly.update_layout(
                title=f"{kpi_description} - Monthly Trend (Last 2 Years)",
                xaxis_title=f"Time ({monthly_granularity}) - Period Column: {time_column}",
                yaxis_title=f"{metric_col} ({aggregation})",
                margin=dict(l=40, r=20, t=50, b=40),
                template="plotly_white",
                height=300,
            )
            
            chart_div_monthly = plot_offline(fig_monthly, include_plotlyjs=False, output_type="div")
            
            html += f'''
            <div style="margin-bottom: 24px;">
                <div style="margin-bottom: 12px; padding: 12px; background: var(--bg); border-radius: 6px; font-size: 13px;">
                    <div><strong>Period Column:</strong> <code>{time_column}</code></div>
                    <div style="margin-top: 4px;"><strong>Formula:</strong> <code>{kpi_formula}</code></div>
                    <div style="margin-top: 4px;"><strong>Granularity:</strong> Monthly (Last 2 Years)</div>
                </div>
                {chart_div_monthly}
            </div>
            '''
        else:
            html += f'''
            <div style="margin-bottom: 24px; padding: 12px; background: var(--bg); border-radius: 6px; font-size: 13px; color: var(--muted);">
                Monthly trend for last 2 years could not be generated (insufficient data in the last 2 years).
            </div>
            '''
        
        html += '</div>'
        return html
    except Exception as e:
        html += f'<p style="color: var(--muted); padding: 12px; background: var(--bg); border-radius: 6px;">Error during time aggregation: {str(e)}</p></div>'
        return html


def _generate_column_correlations(
    column_key: str,
    tests: Dict[str, Any],
    profiles: Dict[str, Any],
) -> str:
    """Generate correlation visualizations for a specific column.
    
    Shows one heatmap per correlated column pair.
    """
    if "." not in column_key:
        return ""
    
    pearson = tests.get("pearson", {})
    if not pearson:
        return ""
    
    # Find correlations involving this column
    correlations = []
    
    # Check if this column is in the correlation matrix
    if column_key in pearson:
        for other_col, corr_value in pearson[column_key].items():
            if other_col != column_key:
                corr_abs = abs(float(corr_value))
                if corr_abs >= 0.3:  # Only show meaningful correlations
                    correlations.append((other_col, float(corr_value)))
    
    # Also check reverse (if other columns reference this one)
    for other_col, row_data in pearson.items():
        if other_col != column_key and column_key in row_data:
            corr_value = row_data[column_key]
            corr_abs = abs(float(corr_value))
            if corr_abs >= 0.3:
                # Avoid duplicates
                if not any(c[0] == other_col for c in correlations):
                    correlations.append((other_col, float(corr_value)))
    
    if not correlations:
        return ""
    
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    html = '<div style="margin-top: 20px;"><h4>Correlations</h4>'
    html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">'
    
    for other_col, corr_value in correlations[:10]:  # Limit to top 10
        # Create 2x2 correlation matrix for this pair
        corr_matrix = [
            [1.0, corr_value],
            [corr_value, 1.0]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=[column_key.split(".")[-1], other_col.split(".")[-1]],
            y=[column_key.split(".")[-1], other_col.split(".")[-1]],
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=[[f"1.00", f"{corr_value:.2f}"], [f"{corr_value:.2f}", "1.00"]],
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title=f"Correlation: {corr_value:.3f}",
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=250,
        )
        
        chart_div = plot_offline(fig, include_plotlyjs=False, output_type="div")
        html += f'<div style="background: var(--card); padding: 12px; border-radius: 8px; border: 1px solid var(--border);">{chart_div}</div>'
    
    html += '</div></div>'
    return html


def _generate_summaries(
    profiles: Dict[str, Any],
    source_name: str,
    table_samples: Optional[Dict[str, pd.DataFrame]] = None,
    tests: Optional[Dict[str, Any]] = None,
    kpi_suggestions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate HTML for the summary views with trend charts and correlations."""
    html = ""
    # Reconstruct hierarchy from profiles
    structure = {}
    for key, profile in profiles.items():
        parts = key.split('.')
        if len(parts) == 2:
            table, column = parts
            if table not in structure:
                structure[table] = {}
            structure[table][column] = profile

    # Table summaries
    for table, columns in structure.items():
        html += f'<div id="summary-{table}" class="summary-view" style="display: none;"><h2>{table} Summary</h2><p>Columns: {len(columns)}</p></div>'

    # Column summaries
    for table, columns in structure.items():
        for column, profile in columns.items():
            column_key = f"{table}.{column}"
            html += f'<div id="summary-{table}-{column}" class="summary-view" style="display: none;">'
            html += f"<h3>{table}.{column} Summary</h3>"
            
            # Display numeric type badge if available
            numeric_type = profile.get("numericType")
            if numeric_type:
                badge_color = "#2e7d32" if numeric_type == "dimension" else "#1976d2"
                html += f'<div style="margin-bottom: 16px;"><span style="background: {badge_color}; color: white; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 12px; text-transform: uppercase;">{numeric_type}</span></div>'
            
            html += "<ul>"
            for stat, value in profile.items():
                # Skip numericType as it's already displayed as badge
                if stat == "numericType":
                    continue
                # Format stats dictionary nicely
                if stat == "stats" and isinstance(value, dict):
                    html += f"<li><strong>{stat}:</strong><ul style='margin-top: 8px; margin-left: 20px;'>"
                    for k, v in value.items():
                        html += f"<li><strong>{k}:</strong> {v}</li>"
                    html += "</ul></li>"
                elif stat == "topK" and isinstance(value, list):
                    html += f"<li><strong>{stat}:</strong><ul style='margin-top: 8px; margin-left: 20px;'>"
                    for item in value[:5]:
                        html += f"<li>{item.get('value', '')}: {item.get('count', 0)}</li>"
                    html += "</ul></li>"
                else:
                    html += f"<li><strong>{stat}:</strong> {value}</li>"
            html += "</ul>"
            
            # Always add trend chart section (will show explanation if cannot generate)
            trend_html = _generate_column_trend_chart(
                column_key, table_samples, kpi_suggestions or [], profile
            )
            html += trend_html
            
            # Add correlations if available
            if tests:
                corr_html = _generate_column_correlations(column_key, tests, profiles)
                html += corr_html
            
            html += "</div>"
    return html


