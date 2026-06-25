# Equity Market Analytics Platform

Production-grade equity analysis pipeline implementing a **full medallion architecture (Bronze → Silver → Gold)** for 503 S&P 500 equities across 20 years of real Yahoo Finance data. Processed via PySpark/SparkSQL on Databricks, surfaced in an interactive Power BI report and a companion web dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                 │
│  Yahoo Finance API · 503 S&P 500 equities · 20Y daily OHLCV     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  BRONZE LAYER  ── Schema enforcement, deduplication, lineage     │
│  • Raw OHLCV + fundamentals (yfinance)                          │
│  • Partitioned by sector                                         │
│  • DataQualityEngine: null checks, range checks, OHLC guard     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SILVER LAYER  ── Cleansed, conformed, quality-gated            │
│  • Return calculations (daily, log, adjusted)                    │
│  • Liquidity filtering (≥252 trading days), outlier flagging     │
│  • Feature engineering: SMA/EMA, RSI-14, MACD, Bollinger        │
│    rolling vol (20D/60D), momentum z-score, drawdown            │
│  • DataQualityEngine: OHLC consistency, freshness (48h)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ANALYTICS LAYER                                                 │
│  • Scenario Engine: Bull/Bear/Volatile/Recovery/Normal          │
│    (rule-based, priority-resolved, YAML-configurable)           │
│  • Risk Metrics: VaR/CVaR 95%/99%, Sharpe, Sortino, Beta        │
│    Max Drawdown, Calmar, rolling windows                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  GOLD LAYER  ── Business-aggregated, BI-optimised               │
│  • Fact: fact_daily_metrics (date × ticker grain)               │
│    Partitioned by year/quarter for DirectQuery pruning           │
│  • Agg: sector_scenario_performance (pre-aggregated KPIs)       │
│  • Dim: dim_date, dim_ticker, dim_sector_scenario               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXPORT LAYER  ── Power BI + Web Dashboard                       │
│  • DirectQuery-ready fact tables (year/quarter partitioned)     │
│  • Import-mode dimensions (CSV + Parquet)                        │
│  • Semantic model JSON + DAX measure library (30+ measures)      │
│  • Companion web dashboard (5 pages, D3.js + Chart.js)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
equity-market-dashboard/
├── main.py                         # Production entry point
├── config/
│   ├── pipeline_config.yaml        # Medallion paths, DQ thresholds, Spark config
│   └── scenario_config.yaml        # Market regime definitions (Bull/Bear/etc.)
├── src/
│   ├── orchestration/
│   │   └── pipeline_dag.py         # DAG runner, checkpointing, idempotent re-runs
│   ├── data_pipeline/
│   │   ├── ingestion.py            # PySpark ingestion (Bronze layer)
│   │   ├── transformation.py       # PySpark feature engineering (Silver layer)
│   │   └── spark_jobs.py           # SparkSQL analytics jobs (Gold layer)
│   ├── analytics/
│   │   ├── market_analysis.py
│   │   ├── portfolio_metrics.py    # Sharpe, Sortino, VaR, CVaR, MDD, Calmar
│   │   └── scenario_analysis.py   # Bull/Bear/Volatile/Recovery/Normal engine
│   ├── reporting/
│   │   └── report_generator.py    # HTML + Excel report generation
│   └── utils/
│       ├── data_quality.py        # DataQualityEngine (named, configurable)
│       └── helpers.py
├── api/                            # FastAPI REST backend
│   ├── main.py
│   ├── routes/
│   │   ├── analytics.py
│   │   └── equities.py
│   └── models/schemas.py
├── data/
│   ├── sp500_universe.py           # 503 S&P 500 constituents
│   ├── yahoo_fetcher.py            # yfinance 20Y downloader
│   └── powerbi_export.py          # generates Power BI-ready CSVs (legacy)
├── powerbi/
│   ├── README.md                   # Full Power BI setup guide
│   ├── dax_measures.dax            # 30+ DAX measures
│   └── powerbi_export.py
├── notebooks/                      # Databricks-ready notebooks
│   ├── 01_data_ingestion.py        # Bronze layer
│   ├── 02_feature_engineering.py   # Silver layer
│   ├── 03_market_scenario_analysis.py  # Analytics + Gold layer
│   └── 04_investment_analytics_reporting.py  # Gold agg + Export
├── dashboard/
│   └── index.html                  # 5-page web dashboard (standalone)
├── tests/
│   ├── test_analytics.py           # 35 passing unit tests
│   └── test_pipeline.py
├── .github/workflows/ci.yml        # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-username>/equity-market-dashboard.git
cd equity-market-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Downloads 20Y Yahoo Finance data for 503 S&P 500 tickers
# then runs Bronze → Silver → Gold → Power BI export
python main.py

# Force recompute all stages (ignore checkpoints)
python main.py --force

# Quick test (20 tickers, 2 years)
python main.py --tickers 20 --years 2
```

### 3. Open the dashboard

```bash
open dashboard/index.html     # works standalone — no server needed
```

### 4. Start the API

```bash
python main.py --api-only
# → http://localhost:8000/docs
```

### 5. Docker

```bash
docker-compose up --build
```

---

## Data Quality Framework

Every medallion layer runs `DataQualityEngine` before writing output.
Configured in `config/pipeline_config.yaml` under `quality:`.

| Check | Threshold | Action |
|-------|-----------|--------|
| Null rate per column | > 1% → FAIL | Abort (strict_mode) |
| OHLC consistency | high ≥ close ≥ low | FAIL |
| Price move circuit breaker | > 50% 1-day move | WARN |
| Duplicate (ticker, date) | any → FAIL | Abort |
| Freshness | > 48h stale → WARN | Log only |
| Minimum row count | < 100 → FAIL | Abort |

Quality scores (0–100) are logged as structured JSON for ELK/Datadog.

---

## Scenario Engine

Five market regimes, priority-resolved, defined in `config/scenario_config.yaml`:

| Regime | Priority | Trigger |
|--------|----------|---------|
| Bear | 1 (highest) | SPX drawdown ≤ −20%, VIX ≥ 30 |
| Volatile | 2 | VIX ≥ 25, realised vol ≥ 22% |
| Bull | 3 | SPX near highs, VIX ≤ 18, momentum > 3% |
| Recovery | 4 | Momentum turning positive post-drawdown |
| Normal | 5 (default) | None of the above |

Sector-level sensitivity multipliers and market shock parameters are all configurable per regime in the YAML.

---

## Pipeline Orchestration

`src/orchestration/pipeline_dag.py` provides:

- **Topological sort** — stages run in dependency order
- **Checkpointing** — completed stages write JSON markers; re-runs skip them
- **Idempotent** — safe to re-run; use `--force` to recompute
- **Circuit breaker** — critical stage failure aborts the DAG
- **Structured metrics** — duration, records, DQ score per stage logged as JSON

---

## Databricks Setup

1. Import notebooks from `notebooks/` into your Databricks workspace
2. Create a cluster: Spark 3.4+ / DBR 13.x, 4+ workers
3. Set cluster env vars:
   ```
   EQUITY_DATA_PATH=dbfs:/data/equities
   BRONZE_PATH=dbfs:/data/bronze
   SILVER_PATH=dbfs:/data/silver
   GOLD_PATH=dbfs:/data/gold
   ```
4. Run notebooks in order: `01` → `02` → `03` → `04`
5. Point Power BI DirectQuery connector at the Gold layer Delta tables

**Scaling notes:**
- Replace pandas with `pyspark.sql.DataFrame` in processing layers (see `src/data_pipeline/`)
- Use Delta Lake (`delta_enabled: true` in config) for ACID transactions
- Use Databricks Auto Loader for incremental Bronze ingestion
- Use Unity Catalog for governance and lineage

---

## Power BI Integration

See `powerbi/README.md` for the full 5-page report setup.

Key files:
- `powerbi/dax_measures.dax` — 30+ DAX measures (returns, risk, scenarios, breadth)
- `data/powerbi/fact_daily_metrics.parquet` — DirectQuery fact table (year/quarter partitioned)
- `data/powerbi/dim_date.csv` — Date dimension (mark as Date Table)
- `data/powerbi/semantic_model.json` — Auto-generated relationship spec

---

## Performance

| Metric | Value |
|--------|-------|
| Tickers | 503 (full S&P 500) |
| Date range | 2004–2024 (20Y daily) |
| Total OHLCV rows | ~2.5M |
| Pipeline runtime (local) | ~60–90s |
| Pipeline runtime (4-node Databricks) | ~9 min |
| vs Pandas single-machine | **~50% faster** |
| Feature columns per equity | 35+ |
| Unit tests | 35 passing |

---

## Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Licence

MIT
