from pathlib import Path
import pandas as pd

from discovery.pipeline.profiler import profile_columns
from discovery.analytics.tests import correlation_matrix
from discovery.insights.ranker import generate_insights
from discovery.render.dashboard import render_dashboard


def test_render_dashboard_writes_html(tmp_path: Path):
    data_path = Path("data/sample_orders.csv").resolve()
    if not data_path.exists():
        return
    
    df = pd.read_csv(data_path)
    cprof = profile_columns(df)
    tests = correlation_matrix(df)
    insights = generate_insights(cprof, tests)
    out_html = tmp_path / "dashboard.html"

    # Test with backward compatible signature (no new optional params)
    render_dashboard("sample_csv", cprof, tests, insights, str(out_html))
    assert out_html.exists()
    html = out_html.read_text(encoding="utf-8")
    assert "Discovery" in html and "Nulls overview" in html
    
    # Test with new optional params
    render_dashboard(
        "sample_csv", cprof, tests, insights, str(tmp_path / "dashboard2.html"),
        relationships=[],
        validation_checks=[],
        kpi_suggestions=[],
        tasks={"tasks": {}, "summary": {"total_tasks": 0}},
    )
    assert (tmp_path / "dashboard2.html").exists()



