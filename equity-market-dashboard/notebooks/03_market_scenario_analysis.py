# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Market Scenario Analysis
# MAGIC
# MAGIC **Purpose:** SparkSQL-first cross-sectional analytics including
# MAGIC sector performance, volatility regime detection, and Bull/Base/Bear
# MAGIC scenario P&L attribution across 500+ equities.
# MAGIC
# MAGIC **Depends on:** `02_feature_engineering.py`

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window

dbutils.widgets.text("FEATURES_PATH", "dbfs:/data/equities/features")
dbutils.widgets.text("OUTPUT_PATH",   "dbfs:/data/equities/analytics")

FEATURES_PATH = dbutils.widgets.get("FEATURES_PATH")
OUTPUT_PATH   = dbutils.widgets.get("OUTPUT_PATH")

# COMMAND ----------

# MAGIC %md ## 1. Load Features & Register SQL View

# COMMAND ----------

features = spark.read.parquet(f"{FEATURES_PATH}/equity_features")
features.createOrReplaceTempView("equity_features")
spark.catalog.cacheTable("equity_features")

print(f"Loaded: {features.count():,} rows | {features.select('ticker').distinct().count()} tickers")

# COMMAND ----------

# MAGIC %md ## 2. Sector Performance Analysis (SparkSQL)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sector-level annualised performance (trailing 252 trading days)
# MAGIC SELECT
# MAGIC     sector,
# MAGIC     COUNT(DISTINCT ticker)                                    AS n_equities,
# MAGIC     ROUND(AVG(daily_return) * 252, 4)                        AS ann_avg_return,
# MAGIC     ROUND(STDDEV(daily_return) * SQRT(252.0), 4)             AS ann_volatility,
# MAGIC     ROUND(
# MAGIC         (AVG(daily_return) * 252 - 0.05) /
# MAGIC         NULLIF(STDDEV(daily_return) * SQRT(252.0), 0), 4
# MAGIC     )                                                         AS sharpe_ratio,
# MAGIC     ROUND(AVG(beta), 3)                                       AS avg_beta,
# MAGIC     ROUND(AVG(pe_ratio), 2)                                   AS avg_pe,
# MAGIC     ROUND(
# MAGIC         SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) /
# MAGIC         COUNT(daily_return), 4
# MAGIC     )                                                         AS win_rate,
# MAGIC     ROUND(MIN(drawdown), 4)                                   AS max_drawdown
# MAGIC FROM equity_features
# MAGIC WHERE date >= DATE_SUB(CURRENT_DATE(), 252)
# MAGIC   AND daily_return IS NOT NULL
# MAGIC GROUP BY sector
# MAGIC ORDER BY ann_avg_return DESC

# COMMAND ----------

sector_perf = spark.sql("""
SELECT
    sector,
    COUNT(DISTINCT ticker)                                    AS n_equities,
    ROUND(AVG(daily_return) * 252, 4)                        AS ann_avg_return,
    ROUND(STDDEV(daily_return) * SQRT(252.0), 4)             AS ann_volatility,
    ROUND(
        (AVG(daily_return) * 252 - 0.05) /
        NULLIF(STDDEV(daily_return) * SQRT(252.0), 0), 4
    )                                                         AS sharpe_ratio,
    ROUND(AVG(beta), 3)                                       AS avg_beta,
    ROUND(MIN(drawdown), 4)                                   AS max_drawdown
FROM equity_features
WHERE date >= DATE_SUB(CURRENT_DATE(), 252)
  AND daily_return IS NOT NULL
GROUP BY sector
ORDER BY ann_avg_return DESC
""")

sector_perf.cache()
display(sector_perf)

# COMMAND ----------

# MAGIC %md ## 3. Top & Bottom Performers

# COMMAND ----------

top_bottom = spark.sql("""
WITH latest AS (
    SELECT ticker, sector, market_cap_category, exchange,
           close, roc_252,
           vol_20d, rsi_14, beta, pe_ratio, dividend_yield,
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
    FROM equity_features
)
SELECT
    ticker,
    sector,
    market_cap_category,
    exchange,
    ROUND(close, 2)        AS latest_close,
    ROUND(roc_252, 4)      AS return_1y,
    ROUND(vol_20d, 4)      AS vol_20d,
    ROUND(rsi_14, 1)       AS rsi_14,
    ROUND(beta, 3)         AS beta,
    ROUND(pe_ratio, 2)     AS pe_ratio,
    ROUND(dividend_yield, 4) AS div_yield
FROM latest
WHERE rn = 1
  AND roc_252 IS NOT NULL
ORDER BY return_1y DESC
""")

print("Top 20 performers:")
display(top_bottom.limit(20))

print("Bottom 20 performers:")
display(top_bottom.orderBy("return_1y").limit(20))

# COMMAND ----------

# MAGIC %md ## 4. Volatility Regime Detection

# COMMAND ----------

vol_regime = spark.sql("""
WITH daily_xsec AS (
    SELECT
        date,
        ROUND(STDDEV(daily_return) * SQRT(252.0), 4) AS cross_sec_vol
    FROM equity_features
    WHERE daily_return IS NOT NULL
    GROUP BY date
),
rolling_vol AS (
    SELECT
        date,
        cross_sec_vol,
        AVG(cross_sec_vol) OVER (
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS rolling_20d_vol
    FROM daily_xsec
)
SELECT
    date,
    cross_sec_vol,
    rolling_20d_vol,
    CASE
        WHEN cross_sec_vol > 0.35 THEN 'High Volatility'
        WHEN cross_sec_vol > 0.20 THEN 'Normal'
        ELSE 'Low Volatility'
    END AS vol_regime
FROM rolling_vol
ORDER BY date
""")

display(vol_regime.tail(30))

# COMMAND ----------

# MAGIC %md ## 5. Market Breadth

# COMMAND ----------

breadth = spark.sql("""
SELECT
    date,
    COUNT(DISTINCT ticker)                                              AS total_tickers,
    SUM(CASE WHEN close > sma_50  THEN 1 ELSE 0 END)                  AS above_sma50,
    SUM(CASE WHEN close > sma_200 THEN 1 ELSE 0 END)                  AS above_sma200,
    SUM(CASE WHEN rsi_14 > 70     THEN 1 ELSE 0 END)                  AS overbought,
    SUM(CASE WHEN rsi_14 < 30     THEN 1 ELSE 0 END)                  AS oversold,
    ROUND(
        SUM(CASE WHEN close > sma_50 THEN 1 ELSE 0 END) /
        NULLIF(COUNT(DISTINCT ticker), 0), 4
    )                                                                    AS breadth_pct
FROM equity_features
WHERE date >= DATE_SUB(CURRENT_DATE(), 252)
  AND sma_50 IS NOT NULL
GROUP BY date
ORDER BY date
""")

display(breadth.tail(20))

# COMMAND ----------

# MAGIC %md ## 6. Scenario Shock Attribution (SparkSQL + PySpark)

# COMMAND ----------

# Scenario assumptions (manually encoded — in production these come from a config table)
SCENARIOS = {
    "bull":  {"mkt_return": 0.25,  "vol_mult": 0.75},
    "base":  {"mkt_return": 0.08,  "vol_mult": 1.00},
    "bear":  {"mkt_return": -0.30, "vol_mult": 1.80},
}

# Compute per-ticker CAPM-implied scenario return for each scenario
latest_betas = spark.sql("""
SELECT ticker, sector, beta
FROM (
    SELECT ticker, sector, beta,
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
    FROM equity_features
    WHERE beta IS NOT NULL
)
WHERE rn = 1
""")

rfr = 0.05
for scen_name, params in SCENARIOS.items():
    mkt_ret = params["mkt_return"]
    latest_betas = latest_betas.withColumn(
        f"return_{scen_name}",
        F.round(rfr + F.col("beta") * (mkt_ret - rfr), 4)
    )

latest_betas = latest_betas.withColumn(
    "upside_capture",
    F.round(F.col("return_bull") / SCENARIOS["bull"]["mkt_return"], 4)
).withColumn(
    "downside_capture",
    F.round(F.col("return_bear") / SCENARIOS["bear"]["mkt_return"], 4)
).withColumn(
    "capture_ratio",
    F.round(F.col("upside_capture") / (F.abs(F.col("downside_capture")) + F.lit(1e-6)), 4)
)

latest_betas.createOrReplaceTempView("scenario_returns")
display(latest_betas.orderBy("capture_ratio", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md ## 7. Sector Scenario Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     sector,
# MAGIC     COUNT(DISTINCT ticker)                   AS n_equities,
# MAGIC     ROUND(AVG(return_bull), 4)               AS avg_bull_return,
# MAGIC     ROUND(AVG(return_base), 4)               AS avg_base_return,
# MAGIC     ROUND(AVG(return_bear), 4)               AS avg_bear_return,
# MAGIC     ROUND(AVG(capture_ratio), 4)             AS avg_capture_ratio,
# MAGIC     ROUND(AVG(beta), 3)                      AS avg_beta
# MAGIC FROM scenario_returns
# MAGIC GROUP BY sector
# MAGIC ORDER BY avg_capture_ratio DESC

# COMMAND ----------

# MAGIC %md ## 8. Persist Analytics Outputs

# COMMAND ----------

for df, name in [
    (sector_perf,   "sector_performance"),
    (top_bottom,    "equity_rankings"),
    (vol_regime,    "volatility_regime"),
    (breadth,       "market_breadth"),
    (latest_betas,  "scenario_returns"),
]:
    path = f"{OUTPUT_PATH}/{name}"
    df.coalesce(1).write.format("parquet").mode("overwrite").save(path)
    print(f"Written -> {path}")

# COMMAND ----------

# MAGIC %md ## Summary
# MAGIC
# MAGIC Outputs written:
# MAGIC - `sector_performance` — 11 sectors, Sharpe/vol/return/beta
# MAGIC - `equity_rankings` — 500 equities ranked by 1Y return
# MAGIC - `volatility_regime` — daily market regime classification
# MAGIC - `market_breadth` — daily % above SMA50/200, RSI extremes
# MAGIC - `scenario_returns` — Bull/Base/Bear CAPM-implied returns, capture ratios
# MAGIC
# MAGIC **Next step:** Run `04_investment_analytics_reporting.py`
