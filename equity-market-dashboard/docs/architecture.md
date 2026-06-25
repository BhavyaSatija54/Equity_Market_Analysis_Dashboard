# Architecture & Data Flow

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
│  Market Data (OHLCV)  │  Fundamentals  │  Index / Benchmarks│
└──────────┬────────────┴───────┬─────────┴──────────┬────────┘
           │                   │                      │
           ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│              DATABRICKS / PYSPARK PIPELINE                   │
│                                                              │
│  01_data_ingestion.py                                        │
│  ├─ Schema enforcement (StructType)                          │
│  ├─ Data quality gates (null %, OHLC consistency)            │
│  └─ Partitioned Parquet output (by sector)                   │
│                                                              │
│  02_feature_engineering.py                                   │
│  ├─ Returns: daily_return, log_return                        │
│  ├─ MAs: SMA 20/50/200, EMA 12/26, MACD                     │
│  ├─ Bollinger Bands (20D, ±2σ)                               │
│  ├─ Momentum: RSI-14, ROC 20/60/252, momentum z-score       │
│  ├─ Volatility: Ann. vol 20D/60D, ATR-14                     │
│  ├─ Cumulative return, drawdown                              │
│  └─ Broadcast join: OHLCV ⋈ Fundamentals                    │
│                                                              │
│  03_market_scenario_analysis.py                              │
│  ├─ SparkSQL: sector performance, top/bottom rankings        │
│  ├─ Volatility regime (High / Normal / Low)                  │
│  ├─ Market breadth (% above SMA50, RSI extremes)             │
│  └─ Scenario CAPM returns (Bull/Base/Bear)                   │
│                                                              │
│  04_investment_analytics_reporting.py                        │
│  ├─ Investment signal scoring (composite: momentum+value+RSI)│
│  ├─ Portfolio attribution by sector                          │
│  └─ HTML + CSV report generation                             │
└──────────────────────────┬──────────────────────────────────┘
                           │  Parquet outputs
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                            │
│                                                              │
│  /api/v1/equities     — screener (500 equities, filters)    │
│  /api/v1/analytics    — risk, scenario, market summary       │
│  /api/v1/reporting    — trigger automated report generation  │
│                                                              │
│  src/analytics/                                              │
│  ├─ portfolio_metrics.py  (Sharpe, Sortino, VaR, CVaR, MDD) │
│  └─ scenario_analysis.py  (Bull/Base/Bear CAPM engine)       │
└──────────────────────────┬──────────────────────────────────┘
                           │  REST JSON
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               INTERACTIVE DASHBOARD                          │
│               dashboard/index.html                           │
│                                                              │
│  ├─ Market Overview   — KPIs, 12M price chart, sector pie    │
│  ├─ Equity Screener   — 500 equities, multi-axis filters     │
│  ├─ Scenario Analysis — Bull/Base/Bear cards, sector chart   │
│  ├─ Risk Metrics      — Sharpe, drawdown, VaR, risk meters   │
│  └─ Sector Breakdown  — heatmap, return/vol comparison       │
│                                                              │
│  Standalone mode: self-contained JS data generation          │
│  API mode:        fetches from FastAPI (localhost:8000)       │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Stage | Input | Output | Runtime (4-node) |
|-------|-------|--------|-----------------|
| Ingestion | ~930K raw rows | ~900K cleansed | ~2 min |
| Feature Engineering | ~900K rows | ~900K × 35 cols | ~4 min |
| Scenario Analysis | ~900K + features | 5 analytics tables | ~2 min |
| Reporting | All analytics | HTML + CSV | ~1 min |
| **Total** | | | **~9 min** |

Workflow acceleration vs. pure Pandas single-machine: **~50%** reduction in wall-clock time due to:
- Partition pruning on sector-partitioned Parquet
- Broadcast join for small fundamentals table
- Adaptive query execution (AQE) for dynamic partition coalescing
- Parallel per-ticker window computation across executors

## Key Design Decisions

**Parquet + partition by sector** — downstream queries on one sector scan only that partition, reducing I/O by ~70–80% for sector-specific jobs.

**Broadcast join** — fundamentals table (~500 rows) is broadcast to all executors, eliminating a full shuffle join that would dominate runtime for the ~900K row OHLCV table.

**SparkSQL views** — analytics queries are written in SQL (notebooks 03/04) for readability, auditability, and easy iteration by quant analysts without Spark expertise.

**Standalone dashboard** — `dashboard/index.html` generates synthetic data in-browser via a seeded RNG, enabling fully offline demos without any backend dependency.
