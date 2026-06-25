# Power BI — Equity Market Analysis Dashboard

## Report Overview

Five report pages analysing 500+ S&P 500 equities across 20 years of daily OHLCV data, built on PySpark/Databricks outputs.

| Page | Description |
|------|-------------|
| 1. Market Intelligence | 20Y S&P 500 history, sector treemap, breadth KPIs |
| 2. Equity Screener | 500+ stock table with sparklines, scatter (return vs vol) |
| 3. Risk Analytics | VaR/CVaR, correlation heatmap, rolling Sharpe, drawdown |
| 4. Scenario Stress Test | Bull/Base/Bear waterfall, radar chart, P&L attribution |
| 5. Sector Rotation | Relative strength quadrant, monthly heatmap |

---

## Data Model

```
OHLCV (fact)
├── ticker       FK → Fundamentals[ticker]
├── date         FK → Dates[date]
├── open, high, low, close, volume
├── daily_return (computed in PySpark: notebook 02)
├── sector       (joined from Fundamentals in PySpark)
└── ... (feature columns: sma_50, rsi_14, vol_20d, etc.)

Fundamentals (dimension)
├── ticker       PK
├── name, sector, sub_industry
├── beta, market_cap_cat, weight_pct
└── pe_ratio, dividend_yield, roe

Dates (date table — mark as Date Table in Power BI)
├── date         PK
├── year, quarter, month_num, month_name
├── week, day_of_week
└── is_trading_day (1/0)

Scenarios (static reference)
├── scenario_name  (Bull / Base / Bear / Stagflation / Rate Shock)
├── mkt_return_shock
└── vol_multiplier
```

**Relationships:**
- OHLCV[ticker] → Fundamentals[ticker]  (Many:1)
- OHLCV[date]   → Dates[date]           (Many:1)

---

## Setup Instructions

### Step 1: Run the PySpark Pipeline

```bash
# 1. Download 20-year data
python data/yahoo_fetcher.py --years 20

# 2. Run Databricks notebooks in order:
#    01_data_ingestion → 02_feature_engineering
#    → 03_market_scenario_analysis → 04_investment_analytics_reporting
```

### Step 2: Export Power BI Data

```bash
python powerbi/powerbi_export.py
# Generates: powerbi/exports/ohlcv_pbi.csv
#            powerbi/exports/fundamentals_pbi.csv
#            powerbi/exports/features_pbi.csv (with RSI, SMA, vol columns)
#            powerbi/exports/scenario_returns.csv
#            powerbi/exports/dates.csv
```

### Step 3: Import into Power BI Desktop

1. Open Power BI Desktop → **Get Data → Text/CSV**
2. Import all five CSV files from `powerbi/exports/`
3. In **Model View**, set relationships (see Data Model above)
4. Mark `dates.csv` as a **Date Table** on the `date` column
5. Right-click each table → **New Measure** → paste from `dax_measures.dax`

### Step 4: Build Report Pages

#### Page 1 — Market Intelligence

| Visual | Type | Fields |
|--------|------|--------|
| S&P 500 20Y Price | Area chart | X: Dates[date], Y: OHLCV[close] filter ticker=SPY |
| Sector Treemap | Treemap | Group: Fundamentals[sector], Size: [Sector Market Cap Weight], Color: [Sector Avg Return 1Y] |
| Market Breadth | Card | [Market Breadth (% Advancing)] |
| VIX Level | Card | (external data source) |
| Top Performers | Table | ticker, [1Y Return], [Annualised Volatility] |

#### Page 2 — Equity Screener

| Visual | Type | Fields |
|--------|------|--------|
| Equity Table | Matrix | Rows: ticker, Cols: sector, Values: [1Y Return], [Annualised Volatility], [Sharpe Ratio] |
| Return vs Vol scatter | Scatter | X: [Annualised Volatility], Y: [1Y Return], Size: weight_pct, Color: sector |
| Sector slicer | Slicer | Fundamentals[sector] |
| Market cap slicer | Slicer | Fundamentals[market_cap_cat] |

**Conditional formatting on table:**
- [1Y Return]: Data bar (Red → White → Green, -50% to +50%)
- [Signal Label]: Background colour rule

#### Page 3 — Risk Analytics

| Visual | Type | Fields |
|--------|------|--------|
| Drawdown | Area chart | X: date, Y: running drawdown column |
| VaR by sector | Bar chart | X: sector, Y: [VaR 95% (1D)], [CVaR 95%] |
| Rolling Sharpe | Line | X: date, Y: [Rolling 252D Sharpe] |
| Risk KPIs | Cards | [Sharpe Ratio], [Max Drawdown], [VaR 95%], [Win Rate] |
| Correlation heatmap | Matrix | Rows: sector_a, Cols: sector_b, Values: correlation (precomputed) |

#### Page 4 — Scenario Stress Test

1. Add **Scenarios** table as a **slicer** (scenario_name field)
2. All scenario measures update automatically via SELECTEDVALUE()

| Visual | Type | Fields |
|--------|------|--------|
| Portfolio return | Card | [Scenario Portfolio Return] |
| Scenario vol | Card | [Scenario Portfolio Vol] |
| Sector waterfall | Waterfall | Category: sector, Values: sector return |
| Scenario compare | Clustered bar | Bull/Base/Bear return by sector |

#### Page 5 — Sector Rotation

| Visual | Type | Fields |
|--------|------|--------|
| RRG quadrant | Scatter | X: [3M Return], Y: [1M Return], Size: weight_pct, Color: sector |
| Monthly heatmap | Matrix | Rows: sector, Cols: month_name, Values: avg monthly return |
| Sector bar | Bar chart | X: sector, Y: [1Y Return] sorted desc |

---

## Slicers (Report-Level)

Add these to the **filter pane** or as page-level slicers:

- `Dates[year]` — year selector (2004–2024)
- `Fundamentals[sector]` — GICS sector
- `Fundamentals[market_cap_cat]` — Large / Mid Cap
- `Scenarios[scenario_name]` — for page 4 only

---

## Power BI Service Publishing

```
1. Power BI Desktop → Publish → select workspace
2. Schedule data refresh:
   - Gateway: Personal Gateway (for local CSV)
   - Or migrate to Azure SQL / Databricks DirectQuery
3. Set refresh: Daily at 6 AM (after market close)
```

---

## DirectQuery via Databricks

For production (no CSV intermediary):

1. In Power BI Desktop: **Get Data → Databricks**
2. Server: `<your-workspace>.azuredatabricks.net`
3. HTTP Path: from cluster settings
4. Select tables: `equity_analytics.ohlcv_cleansed`, `equity_analytics.fundamentals_cleansed`
5. Use **Import mode** for historical data; **DirectQuery** for latest-day snapshot

---

## Performance Tips

- Add `ticker` and `date` as **composite key** in OHLCV
- Enable **Aggregations** on OHLCV for high-cardinality queries
- Pre-aggregate daily → monthly in Power Query for page 5 heatmap
- Use **Calculation groups** for the time intelligence measures (1M/3M/6M/1Y)
- Turn off **Auto date/time** (File → Options → Data Load)
