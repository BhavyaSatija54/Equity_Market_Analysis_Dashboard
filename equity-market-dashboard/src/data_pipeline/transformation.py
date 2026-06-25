"""
src/data_pipeline/transformation.py
-------------------------------------
PySpark feature engineering transformations for equity market data.
All window functions, technical indicators, and risk features computed
here before being served to the analytics layer.

Designed to run efficiently on both Databricks clusters and local PySpark.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from loguru import logger


# ---------------------------------------------------------------------------
# Window specs (reusable)
# ---------------------------------------------------------------------------

def _ticker_window(order_col: str = "date") -> Window:
    """Unbounded window partitioned by ticker, ordered by date."""
    return Window.partitionBy("ticker").orderBy(order_col)


def _ticker_rolling(n: int, order_col: str = "date") -> Window:
    """Rolling window of size n partitioned by ticker."""
    return Window.partitionBy("ticker").orderBy(order_col).rowsBetween(-n + 1, 0)


def _ticker_expanding(order_col: str = "date") -> Window:
    """Expanding window (inception to date) partitioned by ticker."""
    return Window.partitionBy("ticker").orderBy(order_col).rowsBetween(
        Window.unboundedPreceding, 0
    )


# ---------------------------------------------------------------------------
# Returns & log-returns
# ---------------------------------------------------------------------------

def add_returns(df: DataFrame) -> DataFrame:
    """
    Adds daily simple and log returns per ticker.
    Requires columns: ticker, date, close.
    """
    w = _ticker_window()
    df = (
        df
        .withColumn("prev_close", F.lag("close", 1).over(w))
        .withColumn(
            "daily_return",
            (F.col("close") - F.col("prev_close")) / F.col("prev_close")
        )
        .withColumn(
            "log_return",
            F.log(F.col("close") / F.col("prev_close"))
        )
    )
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def add_moving_averages(df: DataFrame, windows: list[int] | None = None) -> DataFrame:
    """SMA and EMA for configurable windows."""
    if windows is None:
        windows = [20, 50, 200]

    for n in windows:
        w_roll = _ticker_rolling(n)
        df = df.withColumn(f"sma_{n}", F.avg("close").over(w_roll))

    # EMA: approximate via exponential smoothing (Spark-native)
    # For production, use a UDF or pandas_udf for exact EMA
    for n in [12, 26]:
        alpha = 2.0 / (n + 1)
        w_exp = _ticker_window()
        # Simplified: use last n-period average as EMA proxy (true EMA needs UDF)
        w_roll = _ticker_rolling(n)
        df = df.withColumn(f"ema_{n}", F.avg("close").over(w_roll))

    # MACD
    df = df.withColumn("macd", F.col("ema_12") - F.col("ema_26"))

    # Bollinger Bands (20-day)
    w20 = _ticker_rolling(20)
    df = (
        df
        .withColumn("bb_mid",   F.avg("close").over(w20))
        .withColumn("bb_std",   F.stddev("close").over(w20))
        .withColumn("bb_upper", F.col("bb_mid") + 2 * F.col("bb_std"))
        .withColumn("bb_lower", F.col("bb_mid") - 2 * F.col("bb_std"))
        .withColumn(
            "bb_position",
            (F.col("close") - F.col("bb_lower")) /
            (F.col("bb_upper") - F.col("bb_lower") + F.lit(1e-8))
        )
    )

    return df


def add_momentum_indicators(df: DataFrame) -> DataFrame:
    """RSI, rate-of-change, and momentum score."""
    w = _ticker_window()

    # Daily gains and losses
    df = (
        df
        .withColumn("gain", F.when(F.col("daily_return") > 0, F.col("daily_return")).otherwise(0.0))
        .withColumn("loss", F.when(F.col("daily_return") < 0, -F.col("daily_return")).otherwise(0.0))
    )

    # 14-day RSI
    w14 = _ticker_rolling(14)
    df = (
        df
        .withColumn("avg_gain", F.avg("gain").over(w14))
        .withColumn("avg_loss", F.avg("loss").over(w14))
        .withColumn(
            "rs", F.col("avg_gain") / (F.col("avg_loss") + F.lit(1e-8))
        )
        .withColumn("rsi_14", 100 - (100 / (1 + F.col("rs"))))
    )

    # Rate of change (20-day, 60-day)
    for n in [20, 60, 252]:
        df = df.withColumn(
            f"roc_{n}",
            (F.col("close") - F.lag("close", n).over(w)) / F.lag("close", n).over(w)
        )

    # Momentum score (z-score of 12M return)
    w252 = _ticker_rolling(252)
    df = df.withColumn("return_252d", F.col("roc_252"))
    df = df.withColumn("momentum_mean", F.avg("roc_252").over(w252))
    df = df.withColumn("momentum_std",  F.stddev("roc_252").over(w252))
    df = df.withColumn(
        "momentum_zscore",
        (F.col("roc_252") - F.col("momentum_mean")) / (F.col("momentum_std") + F.lit(1e-8))
    )

    return df


def add_volatility_features(df: DataFrame) -> DataFrame:
    """Rolling volatility (annualised), Average True Range."""
    for n in [20, 60]:
        w_roll = _ticker_rolling(n)
        df = df.withColumn(
            f"vol_{n}d",
            F.stddev("log_return").over(w_roll) * F.sqrt(F.lit(252))
        )

    # Average True Range (14-day)
    w = _ticker_window()
    df = (
        df
        .withColumn("prev_close_atr", F.lag("close", 1).over(w))
        .withColumn(
            "true_range",
            F.greatest(
                F.col("high") - F.col("low"),
                F.abs(F.col("high") - F.col("prev_close_atr")),
                F.abs(F.col("low")  - F.col("prev_close_atr"))
            )
        )
        .withColumn("atr_14", F.avg("true_range").over(_ticker_rolling(14)))
    )

    return df


def add_volume_features(df: DataFrame) -> DataFrame:
    """Volume ratio and on-balance volume proxy."""
    w = _ticker_window()
    w20 = _ticker_rolling(20)

    df = (
        df
        .withColumn("avg_vol_20d", F.avg("volume").over(w20))
        .withColumn(
            "volume_ratio",
            F.col("volume") / (F.col("avg_vol_20d") + F.lit(1))
        )
        .withColumn(
            "signed_volume",
            F.when(F.col("daily_return") >= 0, F.col("volume"))
             .otherwise(-F.col("volume"))
        )
        .withColumn("obv", F.sum("signed_volume").over(_ticker_expanding()))
    )

    return df


# ---------------------------------------------------------------------------
# Cumulative returns and drawdown
# ---------------------------------------------------------------------------

def add_cumulative_metrics(df: DataFrame) -> DataFrame:
    """Cumulative return (inception to date) and maximum drawdown."""
    w_exp = _ticker_expanding()

    df = (
        df
        .withColumn("cum_return", F.sum("log_return").over(w_exp))
        .withColumn("cum_max_return", F.max("cum_return").over(w_exp))
        .withColumn("drawdown", F.col("cum_return") - F.col("cum_max_return"))
    )

    return df


# ---------------------------------------------------------------------------
# Rolling beta vs. market proxy
# ---------------------------------------------------------------------------

def add_rolling_beta(df: DataFrame, market_proxy_ticker: str = "SPY") -> DataFrame:
    """
    Compute 252-day rolling beta against the market proxy ticker.
    Note: In production, SPY would be a real market index series.
    Here we join on date and use covariance / variance over rolling window.
    """
    # Extract market proxy returns
    market = (
        df
        .filter(F.col("ticker") == market_proxy_ticker)
        .select(F.col("date"), F.col("daily_return").alias("mkt_return"))
    )

    if market.count() == 0:
        logger.warning(
            f"Market proxy '{market_proxy_ticker}' not found in dataset. "
            "Rolling beta will be null."
        )
        return df.withColumn("rolling_beta_252", F.lit(None).cast(T.DoubleType()))

    df = df.join(F.broadcast(market), on="date", how="left")

    w252 = _ticker_rolling(252)
    df = (
        df
        .withColumn("cov_mkt",  F.covar_pop("daily_return", "mkt_return").over(w252))
        .withColumn("var_mkt",  F.var_pop("mkt_return").over(w252))
        .withColumn("rolling_beta_252", F.col("cov_mkt") / (F.col("var_mkt") + F.lit(1e-8)))
        .drop("mkt_return", "cov_mkt", "var_mkt")
    )

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_feature_set(df: DataFrame, include_beta: bool = False) -> DataFrame:
    """
    Run all transformations in dependency order and return enriched DataFrame.
    """
    logger.info("Building feature set")

    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_momentum_indicators(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_cumulative_metrics(df)

    if include_beta:
        df = add_rolling_beta(df)

    # Drop intermediate columns to reduce output size
    intermediate_cols = [
        "gain", "loss", "avg_gain", "avg_loss", "rs",
        "prev_close", "prev_close_atr", "signed_volume",
        "bb_std", "momentum_mean", "momentum_std",
    ]
    existing_intermediates = [c for c in intermediate_cols if c in df.columns]
    df = df.drop(*existing_intermediates)

    logger.info(f"Feature set complete. Columns: {len(df.columns)}")
    return df
