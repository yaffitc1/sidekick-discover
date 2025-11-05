## DataDiscovery

Agentic data exploration for data engineers. Point to a Snowflake schema; the agent profiles your data, probes for issues and strengths, runs statistical queries, and produces dashboards with ranked insights.

### What it does
- **Agentic exploration**: Planner orchestrates tools for schema discovery, sampling, profiling, testing, and visualization.
- **Sources**: First-class support for Snowflake.
- **Profiling & statistics**: Distributions, percentiles, missingness, cardinality, outliers, drift, correlations (Pearson/Spearman/Cramér’s V), and mutual information.
- **Quality checks**: Duplicates, invalids, constraint guesses, schema anomalies, and freshness (where applicable).
- **Insights**: Highlights key signals and risks with natural-language summaries and evidence.
- **Dashboards**: Auto-generated HTML dashboards and notebooks to share findings.
- **Extensible**: Add tools, checks, and connectors as your needs grow.

### Quick start

Prerequisites:
- Python 3.11+
- Snowflake account with credentials configured
- Optional: Docker 24+ for containerized runs

Install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the agent to analyze Snowflake data:

The agent runs locally on your machine and connects to Snowflake to analyze your data. Use Snowflake's public TPCH sample data to try the agent end-to-end.

```bash
# 1) Configure credentials (export here or use a .env file)
export SNOWFLAKE_ACCOUNT=...
export SNOWFLAKE_USER=...
export SNOWFLAKE_PASSWORD=...
export SNOWFLAKE_ROLE=...
export SNOWFLAKE_WAREHOUSE=...

# 2) Register a source (read-only, no table changes)
discovery add snowflake \
  --database SNOWFLAKE_SAMPLE_DATA \
  --schema TPCH_SF1 \
  --tables ORDERS,CUSTOMER \
  --name sf_public

# 3) Explore and generate outputs
discovery agent --source sf_public --sample-rows 100000 --output ./outputs/sf_public

# 4) Open dashboard
discovery dashboard ./outputs/sf_public/dashboard.html
```

### Example: Custom Database

```bash
# Set credentials (or use .env)
export SNOWFLAKE_ACCOUNT=...
export SNOWFLAKE_USER=...
export SNOWFLAKE_PASSWORD=...
export SNOWFLAKE_ROLE=...
export SNOWFLAKE_WAREHOUSE=...
export SNOWFLAKE_DATABASE=...
export SNOWFLAKE_SCHEMA=PUBLIC

# Register a Snowflake source (read-only, no table changes)
discovery add snowflake --database $SNOWFLAKE_DATABASE --schema $SNOWFLAKE_SCHEMA --tables ORDERS,CUSTOMERS --name sf_demo

# Explore and generate dashboards + insights
discovery agent --source sf_demo --sample-rows 100000 --output ./outputs/sf_demo

# Open the dashboard
discovery dashboard ./outputs/sf_demo/dashboard.html
```

### Core commands

- **Register sources**
  ```bash
  discovery add snowflake --database <DB> --schema <SCHEMA> --tables <T1,T2> --name <alias>
  ```

- **Explore with the agent**
  ```bash
  discovery agent --source <alias> [--goal "Find risks and key drivers"] [--sample-rows 100000] --output ./outputs/<alias>
  ```

- **Open dashboards**
  ```bash
  discovery dashboard <path/to/dashboard.html>
  ```

### Configuration

Use environment variables or a `.env` file:
```env
# App
APP_ENV=local
APP_PORT=8000
LOG_LEVEL=info

# Snowflake
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_ROLE=
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_DATABASE=
SNOWFLAKE_SCHEMA=PUBLIC
```

### Outputs

- `outputs/<source>/dashboard.html` — interactive exploration dashboard
- `outputs/<source>/insights.md` — ranked insights with evidence
- `outputs/<source>/profiles/*.json` — column/table profile summaries
- `outputs/<source>/notebooks/*.ipynb` — generated notebooks (optional)

### How it works (high level)

- **Planner (agent)**: Decomposes the goal into steps (sample, profile, test, visualize, rank).
- **Tools**:
  - `schema_discovery`: infer types, date/time, categorical/continuous
  - `sampler`: stratified sampling for stable stats
  - `profiler`: distributions, missingness, cardinality, outliers, drift
  - `hypothesis_tester`: correlations, MI, chi-square, KS tests
  - `visualizer`: histograms, boxplots, scatter/corr heatmaps, time trends
  - `insight_ranker`: scores and explains findings by impact and confidence
- **Renderer**: Builds dashboards and exports notebooks.

### Security & privacy

- Works primarily on metadata, profiles, and samples; avoid exporting full raw data.
- Mask or drop sensitive fields where needed; PII detection is best-effort.
- Use least-privilege Snowflake roles; never store credentials in repos.

### Contributing

- Open issues for proposals/bugs. Ensure tests pass before PRs.
- Style: `ruff` for linting/formatting; `pytest` for tests.

