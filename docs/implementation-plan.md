## Discovery — Implementation Plan (v1)

### Scope and assumptions
- Python 3.11, CLI-first; optional API later.
- Sources: CSV and Snowflake (read-only) in v1.
- Outputs: static HTML dashboards, JSON profiles, Markdown insights under `outputs/<source>/...`.
- No external DB in v1; file-based artifacts only.

### Architecture overview
- `connectors/`: CSV and Snowflake connectors implementing a common interface.
- `pipeline/`: sampling and profiling orchestration per table.
- `analytics/`: statistical tests and association analysis.
- `insights/`: rule-based insight generation and ranking.
- `render/`: Plotly/Jinja dashboard and report rendering.
- `agent/`: planner/orchestrator that executes end-to-end exploration.
- `cli/`: Typer-based CLI entry points.
- `io/`: output paths, JSON/MD/HTML IO helpers.
- `models/`: typed schemas for profiles, tests, insights.

### Project layout
```
discovery/
  agent/
    orchestrator.py
  analytics/
    associations.py
    tests.py
  cli/
    main.py
  connectors/
    csv.py
    snowflake.py
    base.py
  insights/
    ranker.py
    rules.py
  io/
    outputs.py
    files.py
  models/
    types.py
    profiles.py
    insights.py
  pipeline/
    sampler.py
    profiler.py
  render/
    dashboard.py
    templates/
      index.html.j2
README.md
docs/
  implementation-plan.md
  env.example
data/
  sample_orders.csv
```

### Dependencies (v1)
- Data: `pandas`, `numpy`
- Stats/ML: `scipy`, `scikit-learn`, `statsmodels`
- Viz/HTML: `plotly`, `jinja2`
- CLI/Config/Validation: `typer[all]`, `python-dotenv`, `pydantic`
- Snowflake: `snowflake-connector-python`
- Dev: `pytest`, `ruff`

### Phases and deliverables
1) Foundations
- Initialize package structure, config loader, output directory layout.

2) Connectors
- CSV: read (chunked), dtype inference, sample.
- Snowflake: env auth, list tables, schema introspection, limited sampling with `LIMIT` and pushdown filters.
- Common `SourceConnector` interface.

3) Sampling & profiling
- Sampler: head, random, stratified (categorical/time).
- Profiler per table/column:
  - Numeric: count, null%, distinct, min/max, mean/std, p1/p5/p50/p95/p99, outlier flags.
  - Categorical: top-k values, cardinality, entropy.
  - Datetime: min/max, granularity detection, freshness.
  - Quality: duplicates %, negative/zero flags, type invalids.
- Persist to `outputs/<source>/profiles/*.json`.

4) Statistical tests
- Pearson, Spearman; Cramér’s V; mutual information.
- KS (numeric), chi-square (categorical), Shapiro (small n normality).
- Optional drift (recent vs baseline windows).

5) Insights
- Rule engine + ranking by impact (coverage/magnitude), confidence (n, p-values), relevance.
- Export `outputs/<source>/insights.md` (and JSON).

6) Dashboards
- Plotly offline HTML: overview (top insights), per-table visuals (distributions, heatmaps, trends).
- Render via Jinja template → `outputs/<source>/dashboard.html`.

7) Agent orchestrator
- Planner executes: discover schema → sample → profile → tests → visualize → rank → export.
- Checkpointing and resumability; skip steps if artifacts exist and `--force` not set.

8) CLI (Typer)
- `discovery add csv --path <file> --name <alias>`
- `discovery add snowflake --database <DB> --schema <SCHEMA> --tables <T1,T2> --name <alias>`
- `discovery agent run --source <alias> [--goal ...] [--sample-rows N] --output <dir>`
- `discovery insights export --source <alias> --format markdown --out <file>`
- `discovery dashboard open --path <file>`

### Key interfaces (high-level)
```python
# discovery/models/types.py
from typing import Protocol, Iterable, Dict, Optional
import pandas as pd

class SourceConnector(Protocol):
    def list_tables(self) -> Iterable[str]: ...
    def get_schema(self, table: str) -> Dict[str, str]: ...
    def sample(self, table: str, limit: int, method: str = "random",
               stratify_by: Optional[str] = None) -> pd.DataFrame: ...
```

```python
# discovery/pipeline/profiler.py
def profile_table(df) -> dict: ...
def profile_columns(df) -> dict: ...
```

```python
# discovery/analytics/tests.py
def correlation_matrix(df) -> dict: ...
def compute_cramers_v(cat_a, cat_b) -> float: ...
def mutual_information(x, y) -> float: ...
def ks_test(x, y) -> dict: ...
```

```python
# discovery/insights/ranker.py
def generate_insights(profiles: dict, tests: dict) -> list[dict]: ...
```

```python
# discovery/render/dashboard.py
def render_dashboard(source_name: str, profiles: dict, tests: dict,
                     insights: list[dict], out_html: str) -> None: ...
```

```python
# discovery/agent/orchestrator.py
def run(source, output_dir: str, goal: str | None, sample_rows: int) -> None: ...
```

### Data contracts (v1)
- Profiles JSON (per table):
  - `table`, `rowCount`, `duplicatePct`, `columns[]` with `{name, dtype, nullPct, distinct, stats{min,max,mean,std,p1,p5,p50,p95,p99}, topK[], freshness, flags[]}`.
- Tests JSON:
  - `correlations` with Pearson/Spearman/Cramér’s V/MI matrices; include p-values where applicable.
- Insights JSON/MD:
  - `{id, title, severity, score, rationale, evidenceRefs, affectedColumns}`.

### Outputs
- `outputs/<source>/dashboard.html`
- `outputs/<source>/insights.md`
- `outputs/<source>/profiles/*.json`
- `outputs/<source>/notebooks/*.ipynb` (optional)

### Acceptance criteria
- CSV demo produces dashboard and ≥5 insights from `data/sample_orders.csv` in < 2 minutes locally.
- Snowflake demo runs on `SNOWFLAKE_SAMPLE_DATA.TPCH_SF1` (ORDERS, CUSTOMER) with sampling.
- All artifacts render offline; no server required.

### Risks & mitigations
- Snowflake auth/roles: provide `docs/env.example`, document least-privilege and warehouse selection.
- Large tables: enforce `--sample-rows` limits and stratified sampling.
- Statistical noise: include confidence and `n` in scoring; cap pairwise tests by heuristics.

### Non-goals (v1)
- Full lineage extraction, BI tool connectors, persistent metadata store, real-time streaming.

### Next steps
- Scaffold `discovery/` package and CLI.
- Implement CSV connector and minimal profiler.
- Wire the agent to run CSV demo end-to-end.






