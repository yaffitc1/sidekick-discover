from pathlib import Path
import pandas as pd

from discovery.pipeline.profiler import profile_columns
from discovery.analytics.tests import correlation_matrix
from discovery.insights.ranker import generate_insights


def test_generate_insights_returns_list():
    data_path = Path("data/sample_orders.csv").resolve()
    if not data_path.exists():
        return
    
    df = pd.read_csv(data_path)
    cprof = profile_columns(df)
    tests = correlation_matrix(df)
    insights = generate_insights(cprof, tests)
    assert isinstance(insights, list)
    # structure
    if insights:
        first = insights[0]
        assert "title" in first and "score" in first



