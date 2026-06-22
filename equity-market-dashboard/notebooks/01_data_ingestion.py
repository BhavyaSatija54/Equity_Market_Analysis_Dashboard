# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Equity Data Ingestion
# MAGIC
# MAGIC **Purpose:** Load raw equity OHLCV and fundamental data from DBFS/S3/ADLS,
# MAGIC enforce schemas, run data quality checks, and write cleansed Parquet outputs
# MAGIC partitioned by sector for downstream consumption.
# MAGIC
# MAGIC **Runtime:** DBR 13.x | Spark 3.4+ | Python 3.10+
# MAGIC **Cluster:** 4-8 workers, 32GB RAM recommended for full 500-equity universe

# COMMAND ----------

# MAGIC %md ## 0. Setup & Config

# COMMAND ----------

import os
from datetime import date, timedelta

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window

# --- Environment parameters (set as cluster env vars or Databricks widgets) ---
dbutils.widgets.text("EQUITY_DATA_PATH", "dbfs:/data/equities/raw")
dbutils.widgets.text("OUTPUT_PATH",      "dbfs:/data/equities/processed")
dbutils.widgets.text("START_DATE",       str(date.today() - timedelta(days=5*365)))
dbutils.widgets.text("END_DATE",         str(date.today()))
dbutils.widgets.text("MIN_TRADING_DAYS", "252")

EQUITY_DATA_PATH = dbutils.widgets.get("EQUITY_DATA_PATH")
OUTPUT_PATH      = dbutils.widgets.get("OUTPUT_PATH")
START_DATE       = dbutils.widgets.get("START_DATE")
END_DATE         = dbutils.widgets.get("END_DATE")
MIN_TRADING_DAYS = int(dbutils.widgets.get("MIN_TRADING_DAYS"))

print(f"Data path  : {EQUITY_DATA_PATH}")
print(f"Output     : {OUTPUT_PATH}")
print(f"Date range : {START_DATE} -> {END_DATE}")

# COMMAND ----------

# MAGIC %md ## 1. Schema Definitions

# COMMAND ----------

OHLCV_SCHEMA = T.StructType([
    T.StructField("ticker",  T.StringType(),  False),
    T.StructField("date",    T.DateType(),    False),
    T.StructField("open",    T.DoubleType(),  True),
    T.StructField("high",    T.DoubleType(),  True),
    T.StructField("low",     T.DoubleType(),  True),
    T.StructField("close",   T.DoubleType(),  False),
    T.StructField("volume",  T.LongType(),    True),
    T.StructField("sector",  T.StringType(),  True),
])

FUNDAMENTALS_SCHEMA = T.StructType([
    T.StructField("ticker",              T.StringType(),  False),
    T.StructField("sector",              T.StringType(),  True),
    T.StructField("market_cap_category", T.StringType(),  True),
    T.StructField("exchange",            T.StringType(),  True),
    T.StructField("country",             T.StringType(),  True),
    T.StructField("beta",                T.DoubleType(),  True),
    T.StructField("pe_ratio",            T.DoubleType(),  True),
    T.StructField("pb_ratio",            T.DoubleType(),  True),
    T.StructField("dividend_yield",      T.DoubleType(),  True),
    T.StructField("roe",                 T.DoubleType(),  True),
    T.StructField("debt_to_equity",      T.DoubleType(),  True),
])

print("Schemas defined")

# COMMAND ----------

# MAGIC %md ## 2. Read Raw Data

# COMMAND ----------

ohlcv_raw = (
    spark.read
    .format("parquet")
    .schema(OHLCV_SCHEMA)
    .load(f"{EQUITY_DATA_PATH}/ohlcv.parquet")
)

fund_raw = (
    spark.read
    .format("parquet")
    .schema(FUNDAMENTALS_SCHEMA)
    .load(f"{EQUITY_DATA_PATH}/fundamentals.parquet")
)

print(f"OHLCV raw rows      : {ohlcv_raw.count():,}")
print(f"Fundamentals rows   : {fund_raw.count():,}")
print(f"OHLCV schema:")
ohlcv_raw.printSchema()

# COMMAND ----------

# MAGIC %md ## 3. Cleanse OHLCV

# COMMAND ----------

ohlcv_clean = (
    ohlcv_raw
    # Date filter
    .filter(
        (F.col("date") >= F.lit(START_DATE).cast(T.DateType())) &
        (F.col("date") <= F.lit(END_DATE).cast(T.DateType()))
    )
    # Remove null close prices
    .filter(F.col("close").isNotNull())
    # OHLC consistency: high >= close >= low
    .filter(
        (F.col("high") >= F.col("low")) &
        (F.col("close") >= F.col("low") * 0.999) &
        (F.col("close") <= F.col("high") * 1.001)
    )
    # Deduplicate
    .dropDuplicates(["ticker", "date"])
)

# Drop tickers with insufficient history
ticker_day_counts = (
    ohlcv_clean
    .groupBy("ticker")
    .agg(F.count("date").alias("n_days"))
    .filter(F.col("n_days") >= MIN_TRADING_DAYS)
)

ohlcv_clean = ohlcv_clean.join(
    F.broadcast(ticker_day_counts.select("ticker")),
    on="ticker",
    how="inner"
)

# Add metadata
ohlcv_clean = ohlcv_clean.withColumn("_ingested_at", F.current_timestamp())

print(f"OHLCV clean rows    : {ohlcv_clean.count():,}")
print(f"Unique tickers      : {ohlcv_clean.select('ticker').distinct().count()}")
display(ohlcv_clean.limit(10))

# COMMAND ----------

# MAGIC %md ## 4. Cleanse Fundamentals

# COMMAND ----------

fund_clean = (
    fund_raw
    # Cap nonsensical PE ratios
    .withColumn("pe_ratio",
        F.when(F.col("pe_ratio") > 200, None)
         .when(F.col("pe_ratio") < 0,   None)
         .otherwise(F.col("pe_ratio"))
    )
    # Floor dividend yield at 0
    .withColumn("dividend_yield",
        F.when(F.col("dividend_yield") < 0, 0.0)
         .otherwise(F.col("dividend_yield"))
    )
    .dropDuplicates(["ticker"])
    .withColumn("_ingested_at", F.current_timestamp())
)

print(f"Fundamentals clean rows : {fund_clean.count():,}")
display(fund_clean.limit(10))

# COMMAND ----------

# MAGIC %md ## 5. Data Quality Checks

# COMMAND ----------

def run_quality_checks(ohlcv, fundamentals):
    results = {}

    # Null rates
    total    = ohlcv.count()
    null_close = ohlcv.filter(F.col("close").isNull()).count()
    results["null_close_pct"] = null_close / max(total, 1)

    # Ticker coverage
    ohlcv_tickers = ohlcv.select("ticker").distinct()
    fund_tickers  = fundamentals.select("ticker").distinct()
    missing = ohlcv_tickers.join(fund_tickers, on="ticker", how="left_anti")
    results["ohlcv_tickers"]    = ohlcv_tickers.count()
    results["fund_tickers"]     = fund_tickers.count()
    results["missing_fund"]     = missing.count()
    results["ohlcv_rows"]       = total

    # Date coverage
    date_stats = ohlcv.agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date")
    ).collect()[0]
    results["min_date"] = str(date_stats["min_date"])
    results["max_date"] = str(date_stats["max_date"])

    for k, v in results.items():
        print(f"  {k:<30}: {v}")

    # Assert null close below threshold
    assert results["null_close_pct"] < 0.05, \
        f"Null close rate {results['null_close_pct']:.2%} exceeds 5% threshold"

    return results

qc = run_quality_checks(ohlcv_clean, fund_clean)

# COMMAND ----------

# MAGIC %md ## 6. Write Cleansed Outputs (Partitioned Parquet)

# COMMAND ----------

(
    ohlcv_clean
    .repartition(F.col("sector"))
    .write
    .format("parquet")
    .mode("overwrite")
    .partitionBy("sector")
    .save(f"{OUTPUT_PATH}/ohlcv_cleansed")
)

(
    fund_clean
    .coalesce(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"{OUTPUT_PATH}/fundamentals_cleansed")
)

print(f"Written ohlcv_cleansed      -> {OUTPUT_PATH}/ohlcv_cleansed")
print(f"Written fundamentals_cleansed -> {OUTPUT_PATH}/fundamentals_cleansed")

# COMMAND ----------

# MAGIC %md ## 7. Register Delta Tables (optional — requires Delta Lake)

# COMMAND ----------

# Uncomment to register as managed Delta tables for SQL access
# spark.sql(f"""
#   CREATE TABLE IF NOT EXISTS equity_analytics.ohlcv_cleansed
#   USING DELTA
#   LOCATION '{OUTPUT_PATH}/ohlcv_cleansed'
# """)
# spark.sql(f"""
#   CREATE TABLE IF NOT EXISTS equity_analytics.fundamentals_cleansed
#   USING DELTA
#   LOCATION '{OUTPUT_PATH}/fundamentals_cleansed'
# """)
# print("Delta tables registered")

# COMMAND ----------

# MAGIC %md ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Raw OHLCV rows | Loaded |
# MAGIC | Cleansed OHLCV rows | After quality filters |
# MAGIC | Tickers retained | >= MIN_TRADING_DAYS history |
# MAGIC | Output partitioned by | sector |
# MAGIC
# MAGIC **Next step:** Run `02_feature_engineering.py`
