from __future__ import annotations

from typing import Dict, Any, List


def generate_insights(profiles: Dict[str, Any], tests: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a simple set of insights from profiles and correlation tests.

    Heuristics (v1):
    - Flag columns with high missingness.
    - Highlight strongly correlated numeric pairs.
    """
    insights: List[Dict[str, Any]] = []

    # High null percentage
    for col, p in profiles.items():
        null_pct = float(p.get("nullPct", 0.0))
        if null_pct >= 0.2:
            insights.append({
                "id": f"high_null_{col}",
                "title": f"High missingness in {col}",
                "severity": "medium" if null_pct < 0.5 else "high",
                "score": round(null_pct * 100, 2),
                "rationale": f"{null_pct:.1%} nulls detected",
                "affectedColumns": [col],
            })

    # Strong correlations (pearson)
    pearson = tests.get("pearson", {})
    for a, row in pearson.items():
        for b, r in row.items():
            if a >= b:
                continue
            r_abs = abs(float(r))
            if r_abs >= 0.7:
                insights.append({
                    "id": f"corr_{a}_{b}",
                    "title": f"Strong correlation between {a} and {b}",
                    "severity": "medium" if r_abs < 0.9 else "high",
                    "score": round(r_abs * 100, 2),
                    "rationale": f"Pearson r={r_abs:.2f}",
                    "affectedColumns": [a, b],
                })

    # Sort by score desc
    insights.sort(key=lambda x: x.get("score", 0), reverse=True)
    return insights


