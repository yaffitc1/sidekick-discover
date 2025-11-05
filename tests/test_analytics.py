from pathlib import Path
import pandas as pd

from discovery.analytics.tests import correlation_matrix


def test_correlation_matrix_returns_dicts():
    data_path = Path("data/sample_orders.csv").resolve()
    if not data_path.exists():
        return
    
    df = pd.read_csv(data_path)

    tests = correlation_matrix(df)
    assert "pearson" in tests and "spearman" in tests
    # ensure JSON-serializable structure
    assert isinstance(tests["pearson"], dict)



