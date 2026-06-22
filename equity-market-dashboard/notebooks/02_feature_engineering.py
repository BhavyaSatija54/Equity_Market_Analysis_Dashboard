# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature Engineering
# MAGIC
# MAGIC **Purpose:** Compute technical indicators, rolling risk metrics, and
# MAGIC fundamental enrichment via PySpark window functions. Outputs a wide
# MAGIC feature table ready for analytics and ML.
# MAGIC
# MAGIC **Depends on:** `01_data_ingestion.py`

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window

dbutils.widgets.text("PROCESSED_PATH", "dbfs:/data/equities/processed")
dbutils.widgets.text("FEATURES_PATH",  "dbfs:/data/equities/features")

PROCESSED_PATH = dbutils.widgets.get("PROCESSED_PATH")
FEATURES_PATH  = dbutils.widgets.get("FEATURES_PATH")

# COMMAND ----------

# MAGIC %md ## 1. Load Cleansed Data

# COMMAND ----------

ohlcv = spark.read.parquet(f"{PROCESSED_PATH}/ohlcv_cleansed")
fund  = spark.read.parquet(f"{PROCESSED_PATH}/fundamentals_cleansed")

ohlcv.cache()
print(f"OHLCV rows: {ohlcv.count():,} | Tickers: {ohlcv.select('ticker').distinct().count()}")

# COMMAND ----------

# MAGIC %md ## 2. Returns & Log Returns

# COMMAND ----------

w_ticker = Window.partitionBy("ticker").orderBy("date")

ohlcv = (
    ohlcv
    .withColumn("prev_close",   F.lag("close", 1).over(w_ticker))
    .withColumn("daily_return", (F.col("close") - F.col("prev_close")) / F.col("prev_close"))
    .withColumn("log_return",   F.log(F.col("close") / F.col("prev_close")))
    .filter(F.col("daily_return").isNotNull())
)

print("Returns computed")
display(ohlcv.select("ticker", "date", "close", "daily_return", "log_return").limit(10))

# COMMAND ----------

# MAGIC %md ## 3. Moving Averages & Bollinger Bands

# COMMAND ----------

for n in [20, 50, 200]:
    w_roll = Window.partitionBy("ticker").orderBy("date").rowsBetween(-n + 1, 0)
    ohlcv = ohlcv.withColumn(f"sma_{n}", F.avg("close").over(w_roll))

# Bollinger Bands (20-day)
w20 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-19, 0)
ohlcv = (
    ohlcv
    .withColumn("bb_mid",   F.avg("close").over(w20))
    .withColumn("bb_std",   F.stddev("close").over(w20))
    .withColumn("bb_upper", F.col("bb_mid") + 2 * F.col("bb_std"))
    .withColumn("bb_lower", F.col("bb_mid") - 2 * F.col("bb_std"))
    .withColumn(
        "bb_pct",
        (F.col("close") - F.col("bb_lower")) /
        (F.col("bb_upper") - F.col("bb_lower") + F.lit(1e-8))
    )
    .drop("bb_std")
)

print("Moving averages + Bollinger Bands computed")

# COMMAND ----------

# MAGIC %md ## 4. Momentum & RSI

# COMMAND ----------

# Gains and losses for RSI
ohlcv = (
    ohlcv
    .withColumn("gain", F.when(F.col("daily_return") > 0, F.col("daily_return")).otherwise(0.0))
    .withColumn("loss", F.when(F.col("daily_return") < 0, -F.col("daily_return")).otherwise(0.0))
)

w14 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-13, 0)
ohlcv = (
    ohlcv
    .withColumn("avg_gain", F.avg("gain").over(w14))
    .withColumn("avg_loss", F.avg("loss").over(w14))
    .withColumn("rs",       F.col("avg_gain") / (F.col("avg_loss") + F.lit(1e-8)))
    .withColumn("rsi_14",   100 - (100 / (1 + F.col("rs"))))
    .drop("rs", "avg_gain", "avg_loss", "gain", "loss")
)

# Rate of change
for n in [20, 60, 252]:
    ohlcv = ohlcv.withColumn(
        f"roc_{n}",
        (F.col("close") - F.lag("close", n).over(w_ticker)) /
        F.lag("close", n).over(w_ticker)
    )

# Momentum z-score (252-day ROC)
w252 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-251, 0)
ohlcv = (
    ohlcv
    .withColumn("mom_mu",  F.avg("roc_252").over(w252))
    .withColumn("mom_std", F.stddev("roc_252").over(w252))
    .withColumn(
        "momentum_zscore",
        (F.col("roc_252") - F.col("mom_mu")) / (F.col("mom_std") + F.lit(1e-8))
    )
    .drop("mom_mu", "mom_std")
)

print("Momentum indicators computed")

# COMMAND ----------

# MAGIC %md ## 5. Volatility Features

# COMMAND ----------

for n in [20, 60]:
    w_vol = Window.partitionBy("ticker").orderBy("date").rowsBetween(-n + 1, 0)
    ohlcv = ohlcv.withColumn(
        f"vol_{n}d",
        F.stddev("log_return").over(w_vol) * F.sqrt(F.lit(252))
    )

# ATR (14-day)
ohlcv = (
    ohlcv
    .withColumn("prev_close_atr", F.lag("close", 1).over(w_ticker))
    .withColumn(
        "true_range",
        F.greatest(
            F.col("high") - F.col("low"),
            F.abs(F.col("high") - F.col("prev_close_atr")),
            F.abs(F.col("low")  - F.col("prev_close_atr"))
        )
    )
    .withColumn(
        "atr_14",
        F.avg("true_range").over(
            Window.partitionBy("ticker").orderBy("date").rowsBetween(-13, 0)
        )
    )
    .drop("prev_close_atr", "true_range")
)

print("Volatility features computed")

# COMMAND ----------

# MAGIC %md ## 6. Cumulative Return & Drawdown

# COMMAND ----------

w_exp = Window.partitionBy("ticker").orderBy("date").rowsBetween(Window.unboundedPreceding, 0)

ohlcv = (
    ohlcv
    .withColumn("cum_log_return", F.sum("log_return").over(w_exp))
    .withColumn("cum_max",        F.max("cum_log_return").over(w_exp))
    .withColumn("drawdown",       F.col("cum_log_return") - F.col("cum_max"))
    .drop("cum_max")
)

print("Cumulative metrics computed")

# COMMAND ----------

# MAGIC %md ## 7. Enrich with Fundamentals (Broadcast Join)

# COMMAND ----------

fund_subset = fund.select(
    "ticker", "sector", "market_cap_category", "exchange",
    "country", "beta", "pe_ratio", "dividend_yield", "roe"
)

# Broadcast join: fundamentals is small enough to broadcast
features = ohlcv.join(F.broadcast(fund_subset), on="ticker", how="left")

print(f"Feature table: {features.count():,} rows | {len(features.columns)} columns")
display(features.limit(5))

# COMMAND ----------

# MAGIC %md ## 8. MA Signal Labels

# COMMAND ----------

features = (
    features
    .withColumn(
        "ma_signal",
        F.when(F.col("sma_50") > F.col("sma_200"), "Golden Cross")
         .otherwise("Death Cross")
    )
    .withColumn(
        "rsi_signal",
        F.when(F.col("rsi_14") > 70, "Overbought")
         .when(F.col("rsi_14") < 30, "Oversold")
         .otherwise("Neutral")
    )
)

# COMMAND ----------

# MAGIC %md ## 9. Write Feature Table

# COMMAND ----------

(
    features
    .repartition(F.col("sector"))
    .write
    .format("parquet")
    .mode("overwrite")
    .partitionBy("sector")
    .save(f"{FEATURES_PATH}/equity_features")
)

# Register as a temp view for downstream SparkSQL notebooks
features.createOrReplaceTempView("equity_features")
spark.catalog.cacheTable("equity_features")

print(f"Written -> {FEATURES_PATH}/equity_features")
print(f"Columns: {features.columns}")

# COMMAND ----------

# MAGIC %md ## Summary
# MAGIC
# MAGIC Feature set includes:
# MAGIC - **Price**: open, high, low, close, volume
# MAGIC - **Returns**: daily_return, log_return, roc_20/60/252
# MAGIC - **Moving Averages**: sma_20, sma_50, sma_200, ema_12, ema_26
# MAGIC - **Momentum**: rsi_14, momentum_zscore, ma_signal, rsi_signal
# MAGIC - **Volatility**: vol_20d, vol_60d, atr_14, bb_upper/lower/pct
# MAGIC - **Cumulative**: cum_log_return, drawdown
# MAGIC - **Fundamentals**: beta, pe_ratio, dividend_yield, roe, sector, market_cap_category
# MAGIC
# MAGIC **Next step:** Run `03_market_scenario_analysis.py`
