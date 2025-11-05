from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import plotly.graph_objs as go
from plotly.offline import plot as plot_offline


def _table_overview(profiles: Dict[str, Any]) -> go.Figure:
    """Bar chart of null percentage per column for quick data hygiene view."""
    cols = list(profiles.keys())
    nulls = [round(float(profiles[c].get("nullPct", 0.0)) * 100, 2) for c in cols]
    fig = go.Figure(data=[go.Bar(x=cols, y=nulls, marker_color="#2e7d32")])
    fig.update_layout(
        title="Null percentage by column",
        xaxis_title="Column",
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


def _relationship_graph(relationship_graph: Optional[Dict[str, Any]]) -> str:
    """Generate HTML for relationship graph visualization."""
    if not relationship_graph or not relationship_graph.get("edges"):
        return '<p style="color: var(--muted);">No relationships detected.</p>'
    
    # Simple HTML representation (can be enhanced with D3.js or Plotly network graph)
    edges = relationship_graph.get("edges", [])
    nodes = relationship_graph.get("nodes", [])
    
    html = f'<div style="margin: 10px 0;"><p><strong>{len(nodes)} tables</strong> with <strong>{len(edges)} relationships</strong></p>'
    html += '<div style="max-height: 400px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; padding: 12px;">'
    
    for edge in edges[:20]:  # Limit to top 20
        source = edge.get("source", "unknown")
        target = edge.get("target", "unknown")
        match_rate = edge.get("match_rate", 0)
        confidence = edge.get("confidence", "unknown")
        
        html += f'''
        <div style="padding: 8px; margin-bottom: 8px; background: var(--bg); border-radius: 6px;">
          <strong>{source}</strong> → <strong>{target}</strong><br>
          <span style="font-size: 12px; color: var(--muted);">Match: {match_rate:.1%} | Confidence: {confidence}</span>
        </div>
        '''
    
    if len(edges) > 20:
        html += f'<p style="font-size: 12px; color: var(--muted);">... and {len(edges) - 20} more relationships</p>'
    
    html += '</div></div>'
    return html


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
) -> None:
    """Render an offline HTML dashboard with a green, clean theme.
    
    Enhanced with relationships, validation, KPIs, and actionable tasks sections.
    """
    out_path = Path(out_html).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    
    relationships_html = _relationship_graph(relationship_graph)
    validation_html = _validation_summary(validation_checks)
    kpi_html = _kpi_suggestions(kpi_suggestions)
    tasks_html = _actionable_tasks(tasks)
    
    # Build section HTML
    overview_section = f'<section id="overview" class="section grid"><div class="card" style="grid-column: 1 / -1;"><div class="head"><h2>Overview</h2></div><div class="body">{overview_html}</div></div></section>' if overview_html else ''
    relationships_section = f'<section id="relationships" class="section grid"><div class="card" style="grid-column: 1 / -1;"><div class="head"><h2>Relationships</h2></div><div class="body">{relationships_html}</div></div></section>' if relationships_html else ''
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
      }}
      .header {{
        background: linear-gradient(135deg, var(--brand) 0%, var(--brand-300) 100%);
        color: #fff;
        padding: 24px 20px 48px;
      }}
      .nav {{
        display: flex; align-items: center; justify-content: space-between;
        max-width: 1120px; margin: 0 auto;
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
      .anchors a {{ color: #fff; opacity: 0.95; }}
      .anchors a:hover {{ opacity: 1; text-decoration: underline; }}
    </style>
  </head>
  <body>
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
          <a href="#overview">Overview</a>
          <a href="#insights">Insights</a>
          <a href="#relationships">Relationships</a>
          <a href="#validation">Validation</a>
          <a href="#kpis">KPIs</a>
          <a href="#tasks">Tasks</a>
          <a href="#nulls">Nulls</a>
          <a href="#correlations">Correlations</a>
        </nav>
      </div>
      <div class="hero">
        <h1>Source: {source_name}</h1>
        <p>Automated profiles, statistical tests and ranked insights.</p>
      </div>
    </header>

    <main class="container">
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

      <section id="correlations" class="section grid">
        <div class="card" style="grid-column: 1 / -1;">
          <div class="head"><h2>Correlation heatmap</h2></div>
          <div class="body">{corr_div if corr_div else '<p style="color: var(--muted);">No numeric columns available for correlations.</p>'}</div>
        </div>
      </section>
    </main>

    <footer class="footer">Generated by Discovery</footer>
  </body>
 </html>
"""
    out_path.write_text(html, encoding="utf-8")


