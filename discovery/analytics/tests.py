from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


def correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlations for numeric columns.

    Returns nested dicts keyed by column name to ensure JSON serializable output.
    """
    # Exclude boolean dtype to avoid issues with boolean arithmetic
    num_df = df.select_dtypes(include=[np.number], exclude=['bool'])
    if num_df.shape[1] == 0:
        return {"pearson": {}, "spearman": {}}
    pearson = num_df.corr(method="pearson").fillna(0.0)
    spearman = num_df.corr(method="spearman").fillna(0.0)
    return {
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
    }


