# Equity Market Analysis Dashboard

An end-to-end equity analytics platform analysing 500+ equities across market scenarios. Combines PySpark/SparkSQL data pipelines (Databricks-native) with a FastAPI backend and an interactive web dashboard for automated investment analytics and reporting.

---

## Overview

| Layer | Technology |
|---|---|
| Data Pipeline | PySpark, SparkSQL, Databricks |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Analytics Engine | NumPy, Pandas, SciPy, scikit-learn |
| Dashboard | Vanilla JS, Chart.js, Tabulator |
| Testing | pytest, pytest-asyncio |
| CI/CD | GitHub Actions |
| Containerisation | Docker, Docker Compose |

---

## Features

- **500+ Equity Screener** — filter by sector, market cap, beta, momentum, and custom signals
- **Market Scenario Analysis** — Bull / Base / Bear case stress testing with VaR and CVaR
- **Risk Metrics Engine** — Sharpe, Sortino, Treynor, Max Drawdown, Rolling Beta, GARCH volatility
- **PySpark ETL Pipeline** — batch-optimised ingestion and transformation for large-scale financial datasets
- **SparkSQL Analytics** — SQL-first investment analytics with reusable query templates
- **Automated Reporting** — PDF/HTML report generation with performance attribution
- **Databricks Notebooks** — production-grade notebooks for each pipeline stage
- **REST API** — all analytics exposed as typed FastAPI endpoints with OpenAPI docs

---

## Project Structure

```
equity-market-dashboard/
├── api/                        # FastAPI application
│   ├── main.py                 # App entry point, CORS, middleware
│   ├── models/schemas.py       # Pydantic request/response models
│   └── routes/
│       ├── analytics.py        # Risk, scenario, portfolio endpoints
│       └── equities.py         # Equity screener endpoints
├── config/
│   └── config.yaml             # Environment-aware configuration
├── dashboard/
│   └── index.html              # Standalone interactive dashboard
├── data/
│   └── sample_generator.py     # Generates realistic 500-equity dataset
├── docs/
│   └── architecture.md         # System design and data flow
├── notebooks/                  # Databricks-ready notebooks (Python format)
│   ├── 01_data_ingestion.py
│   ├── 02_feature_engineering.py
│   ├── 03_market_scenario_analysis.py
│   └── 04_investment_analytics_reporting.py
├── src/
│   ├── analytics/
│   │   ├── market_analysis.py  # Core market analytics
│   │   ├── portfolio_metrics.py
│   │   └── scenario_analysis.py
│   ├── data_pipeline/
│   │   ├── ingestion.py        # Source connectors
│   │   ├── transformation.py   # Feature engineering
│   │   └── spark_jobs.py       # Spark job orchestration
│   ├── reporting/
│   │   └── report_generator.py
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_analytics.py
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/equity-market-dashboard.git
cd equity-market-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate sample data

```bash
python data/sample_generator.py
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 4. Open the dashboard

Open `dashboard/index.html` directly in a browser — it works standalone with built-in generated data, or connects to the API if running.

### 5. Docker (full stack)

```bash
docker-compose up --build
```

---

## Databricks Setup

1. Import notebooks from `notebooks/` into your Databricks workspace
2. Create a cluster with Spark 3.4+ / DBR 13.x
3. Set the following environment variables in your cluster config:

```
EQUITY_DATA_PATH=dbfs:/data/equities
OUTPUT_PATH=dbfs:/data/output
ENVIRONMENT=databricks
```

4. Run notebooks in order: `01` → `02` → `03` → `04`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/equities` | List equities with filters |
| GET | `/api/v1/equities/{ticker}` | Single equity detail |
| POST | `/api/v1/analytics/risk` | Portfolio risk metrics |
| POST | `/api/v1/analytics/scenario` | Scenario analysis (Bull/Base/Bear) |
| GET | `/api/v1/analytics/market-summary` | Aggregated market overview |
| POST | `/api/v1/reporting/generate` | Trigger automated report |

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Performance Notes

- PySpark transformations use partition pruning and broadcast joins to handle datasets of 500+ equities with 5+ years of daily OHLCV data (~900k+ rows)
- SparkSQL views are cached for repeated analytical queries
- API responses are cached with a 60-second TTL for market summary endpoints
- Dashboard renders 500 equities in the screener table with virtualised rows (no DOM thrashing)

---

## Licence

MIT
