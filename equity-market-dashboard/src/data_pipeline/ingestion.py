"""
src/data_pipeline/ingestion.py
-------------------------------
PySpark-based ingestion layer for equity market data.
Handles batch reads from Parquet/Delta/CSV sources with schema enforcement,
data quality checks, and partition-aware loading.

Compatible with Databricks Runtime 13.x and local PySpark 3.5.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

OHLCV_SCHEMA = T.StructType([
    T.StructField("ticker",  T.StringType(),    nullable=False),
    T.StructField("date",    T.DateType(),       nullable=False),
    T.StructField("open",    T.DoubleType(),     nullable=True),
    T.StructField("high",    T.DoubleType(),     nullable=True),
    T.StructField("low",     T.DoubleType(),     nullable=True),
    T.StructField("close",   T.DoubleType(),     nullable=False),
    T.StructField("volume",  T.LongType(),       nullable=True),
    T.StructField("sector",  T.StringType(),     nullable=True),
])

FUNDAMENTALS_SCHEMA = T.StructType([
    T.StructField("ticker",              T.StringType(),  nullable=False),
    T.StructField("sector",              T.StringType(),  nullable=True),
    T.StructField("market_cap_category", T.StringType(),  nullable=True),
    T.StructField("exchange",            T.StringType(),  nullable=True),
    T.StructField("country",             T.StringType(),  nullable=True),
    T.StructField("beta",                T.DoubleType(),  nullable=True),
    T.StructField("pe_ratio",            T.DoubleType(),  nullable=True),
    T.StructField("pb_ratio",            T.DoubleType(),  nullable=True),
    T.StructField("dividend_yield",      T.DoubleType(),  nullable=True),
    T.StructField("roe",                 T.DoubleType(),  nullable=True),
    T.StructField("debt_to_equity",      T.DoubleType(),  nullable=True),
])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IngestionConfig:
    ohlcv_path: str = "data/raw/ohlcv.parquet"
    fundamentals_path: str = "data/raw/fundamentals.parquet"
    output_path: str = "data/processed"
    source_format: str = "parquet"          # parquet | delta | csv
    start_date: Optional[str] = None        # YYYY-MM-DD filter
    end_date: Optional[str] = None
    tickers: list[str] = field(default_factory=list)   # empty = all
    min_trading_days: int = 252             # drop tickers with < N days
    max_null_pct: float = 0.05             # fail if > 5% nulls on close


# ---------------------------------------------------------------------------
# Ingestion class
# ---------------------------------------------------------------------------

class EquityDataIngestion:
    """
    Loads raw equity data from Parquet/Delta/CSV, enforces schemas,
    runs quality gates, and writes cleansed partitioned output.
    """

    def __init__(self, spark: SparkSession, config: IngestionConfig | None = None):
        self.spark = spark
        self.cfg = config or IngestionConfig()
        self._quality_report: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, DataFrame]:
        """Full ingestion pipeline. Returns dict of cleansed DataFrames."""
        logger.info("Starting equity data ingestion")

        ohlcv_raw = self._read(self.cfg.ohlcv_path, OHLCV_SCHEMA)
        fund_raw  = self._read(self.cfg.fundamentals_path, FUNDAMENTALS_SCHEMA)

        ohlcv_clean = self._cleanse_ohlcv(ohlcv_raw)
        fund_clean  = self._cleanse_fundamentals(fund_raw)

        self._run_quality_checks(ohlcv_clean, fund_clean)

        self._write(ohlcv_clean, "ohlcv_cleansed", partition_by=["sector"])
        self._write(fund_clean,  "fundamentals_cleansed")

        logger.info(
            f"Ingestion complete. "
            f"OHLCV: {ohlcv_clean.count():,} rows | "
            f"Fundamentals: {fund_clean.count():,} rows"
        )
        return {"ohlcv": ohlcv_clean, "fundamentals": fund_clean}

    def quality_report(self) -> dict:
        return self._quality_report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read(self, path: str, schema: T.StructType) -> DataFrame:
        fmt = self.cfg.source_format
        logger.debug(f"Reading [{fmt}] from {path}")
        reader = self.spark.read.format(fmt)
        if fmt == "csv":
            reader = reader.option("header", "true").option("inferSchema", "false")
            df = reader.schema(schema).load(path)
        else:
            df = reader.load(path)
            # Cast to expected schema (Delta may have evolved schema)
            df = self.spark.createDataFrame(df.rdd, schema)
        return df

    def _cleanse_ohlcv(self, df: DataFrame) -> DataFrame:
        logger.debug("Cleansing OHLCV")

        # Date filters
        if self.cfg.start_date:
            df = df.filter(F.col("date") >= F.lit(self.cfg.start_date).cast(T.DateType()))
        if self.cfg.end_date:
            df = df.filter(F.col("date") <= F.lit(self.cfg.end_date).cast(T.DateType()))

        # Ticker filter
        if self.cfg.tickers:
            df = df.filter(F.col("ticker").isin(self.cfg.tickers))

        # Drop rows with null close
        df = df.filter(F.col("close").isNotNull())

        # Sanity checks: high >= low, close within [low, high]
        df = df.filter(
            (F.col("high") >= F.col("low")) &
            (F.col("close") >= F.col("low")) &
            (F.col("close") <= F.col("high") * 1.001)  # small tolerance
        )

        # Drop tickers with insufficient history
        trading_days = (
            df.groupBy("ticker")
              .agg(F.count("date").alias("n_days"))
              .filter(F.col("n_days") >= self.cfg.min_trading_days)
        )
        df = df.join(F.broadcast(trading_days.select("ticker")), on="ticker", how="inner")

        # Deduplicate (ticker, date)
        df = df.dropDuplicates(["ticker", "date"])

        # Add ingestion metadata
        df = df.withColumn("_ingested_at", F.current_timestamp())

        return df.orderBy("ticker", "date")

    def _cleanse_fundamentals(self, df: DataFrame) -> DataFrame:
        logger.debug("Cleansing fundamentals")

        if self.cfg.tickers:
            df = df.filter(F.col("ticker").isin(self.cfg.tickers))

        # Cap extreme PE ratios (data quality artefact)
        df = df.withColumn(
            "pe_ratio",
            F.when(F.col("pe_ratio") > 200, None).otherwise(F.col("pe_ratio"))
        )
        # Negative dividend yield is meaningless
        df = df.withColumn(
            "dividend_yield",
            F.when(F.col("dividend_yield") < 0, 0.0).otherwise(F.col("dividend_yield"))
        )

        df = df.dropDuplicates(["ticker"])
        df = df.withColumn("_ingested_at", F.current_timestamp())

        return df

    def _run_quality_checks(self, ohlcv: DataFrame, fundamentals: DataFrame) -> None:
        logger.info("Running data quality checks")
        n_ohlcv = ohlcv.count()
        n_fund  = fundamentals.count()

        # Null rate on close price
        null_close = ohlcv.filter(F.col("close").isNull()).count()
        null_pct   = null_close / max(n_ohlcv, 1)
        if null_pct > self.cfg.max_null_pct:
            raise ValueError(
                f"Null close rate {null_pct:.2%} exceeds threshold {self.cfg.max_null_pct:.2%}"
            )

        # Check ticker coverage
        ohlcv_tickers = ohlcv.select("ticker").distinct().count()
        fund_tickers  = fundamentals.select("ticker").distinct().count()
        missing_fund  = ohlcv.select("ticker").distinct() \
                             .join(fundamentals.select("ticker").distinct(),
                                   on="ticker", how="left_anti")

        self._quality_report = {
            "ohlcv_rows":        n_ohlcv,
            "ohlcv_tickers":     ohlcv_tickers,
            "fundamental_tickers": fund_tickers,
            "null_close_pct":    null_pct,
            "tickers_missing_fundamentals": missing_fund.count(),
        }

        logger.info(
            f"QA | rows={n_ohlcv:,} | tickers={ohlcv_tickers} | "
            f"null_close={null_pct:.3%} | missing_fund={self._quality_report['tickers_missing_fundamentals']}"
        )

    def _write(self, df: DataFrame, name: str, partition_by: list[str] | None = None) -> None:
        path = os.path.join(self.cfg.output_path, name)
        writer = df.write.format("parquet").mode("overwrite")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)
        logger.info(f"Written -> {path}")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_spark_session(app_name: str = "EquityIngestion", local: bool = True) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
    )
    if local:
        builder = builder.master("local[*]")

    # Delta Lake support (optional; skip if not installed)
    try:
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
    except Exception:
        pass

    return builder.getOrCreate()
