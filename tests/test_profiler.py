from pathlib import Path
import pandas as pd

from discovery.pipeline.profiler import profile_table, profile_columns


def test_profile_table_and_columns():
    data_path = Path("data/sample_orders.csv").resolve()
    if not data_path.exists():
        return
    
    df = pd.read_csv(data_path)

    tprof = profile_table(df)
    cprof = profile_columns(df)

    assert tprof["rowCount"] == len(df)
    assert "amount" in cprof
    assert "nullPct" in cprof["amount"]
    # numeric stats present
    assert "stats" in cprof["amount"]



