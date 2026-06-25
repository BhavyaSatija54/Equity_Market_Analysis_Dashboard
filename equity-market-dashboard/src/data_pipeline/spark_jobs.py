"""
src/data_pipeline/spark_jobs.py
---------------------------------
SparkSQL-first investment analytics job templates.
These can be registered as Databricks jobs or run standalone.

SparkSQL views are used to enable SQL-first analytical queries
that business users can inspect and iterate on.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from loguru import logger


# ---------------------------------------------------------------------------
# View registration
# ---------------------------------------------------------------------------

def register_views(spark: SparkSession, ohlcv: DataFrame, fundamentals: DataFrame) -> None:
    """Register temp views so SparkSQL queries work directly."""
    ohlcv.createOrReplaceTempView("ohlcv")
    fundamentals.createOrReplaceTempView("fundamentals")
    logger.info("Registered SparkSQL views: ohlcv, fundamentals")


# ---------------------------------------------------------------------------
# SparkSQL analytics queries
# ---------------------------------------------------------------------------

SECTOR_PERFORMANCE_SQL = """
SELECT
    f.sector,
    COUNT(DISTINCT o.ticker)                                    AS n_equities,
    ROUND(AVG(o.daily_return) * 252, 4)                        AS annualised_avg_return,
    ROUND(STDDEV(o.daily_return) * SQRT(252), 4)               AS annualised_volatility,
    ROUND(
        (AVG(o.daily_return) * 252 - 0.05) /
        NULLIF(STDDEV(o.daily_return) * SQRT(252), 0), 4
    )                                                           AS sharpe_ratio,
    ROUND(AVG(f.beta), 3)                                      AS avg_beta,
    ROUND(AVG(f.pe_ratio), 2)                                  AS avg_pe,
    ROUND(SUM(CASE WHEN o.daily_return > 0 THEN 1 ELSE 0 END) /
          COUNT(o.daily_return), 4)                             AS win_rate
FROM ohlcv o
JOIN fundamentals f ON o.ticker = f.ticker
WHERE o.date >= DATE_SUB(CURRENT_DATE(), 252)
GROUP BY f.sector
ORDER BY annualised_avg_return DESC
"""

TOP_PERFORMERS_SQL = """
WITH latest AS (
    SELECT
        ticker,
        close,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
    FROM ohlcv
),
one_year_ago AS (
    SELECT
        ticker,
        close AS close_1y,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date ASC) AS rn
    FROM ohlcv
    WHERE date >= DATE_SUB(CURRENT_DATE(), 252)
),
perf AS (
    SELECT
        l.ticker,
        l.close AS latest_close,
        a.close_1y,
        ROUND((l.close - a.close_1y) / NULLIF(a.close_1y, 0), 4) AS return_1y
    FROM latest l
    JOIN one_year_ago a ON l.ticker = a.ticker
    WHERE l.rn = 1 AND a.rn = 1
)
SELECT
    p.ticker,
    f.sector,
    f.market_cap_category,
    f.exchange,
    p.latest_close,
    p.return_1y,
    f.beta,
    f.pe_ratio,
    f.dividend_yield
FROM perf p
JOIN fundamentals f ON p.ticker = f.ticker
ORDER BY p.return_1y DESC
LIMIT 50
"""

VOLATILITY_REGIME_SQL = """
WITH daily_stats AS (
    SELECT
        date,
        STDDEV(daily_return) * SQRT(252) AS cross_sectional_vol
    FROM ohlcv
    WHERE daily_return IS NOT NULL
    GROUP BY date
),
regime AS (
    SELECT
        date,
        cross_sectional_vol,
        AVG(cross_sectional_vol) OVER (
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS rolling_20d_vol,
        CASE
            WHEN cross_sectional_vol >  0.35 THEN 'High Volatility'
            WHEN cross_sectional_vol >= 0.20 THEN 'Normal'
            ELSE 'Low Volatility'
        END AS vol_regime
    FROM daily_stats
)
SELECT * FROM regime ORDER BY date
"""

ROLLING_RISK_METRICS_SQL = """
SELECT
    ticker,
    date,
    daily_return,
    vol_20d,
    vol_60d,
    rsi_14,
    roc_20,
    momentum_zscore,
    sma_50,
    sma_200,
    CASE WHEN sma_50 > sma_200 THEN 'Golden Cross' ELSE 'Death Cross' END AS ma_signal,
    CASE
        WHEN rsi_14 > 70 THEN 'Overbought'
        WHEN rsi_14 < 30 THEN 'Oversold'
        ELSE 'Neutral'
    END AS rsi_signal
FROM ohlcv_features
WHERE date >= DATE_SUB(CURRENT_DATE(), 252)
"""

CORRELATION_MATRIX_SQL = """
-- Pair-wise correlation proxy (same-day return correlation across all tickers)
-- In production this would use a proper correlation UDF or pivot approach
SELECT
    a.ticker   AS ticker_a,
    b.ticker   AS ticker_b,
    ROUND(CORR(a.daily_return, b.daily_return), 4) AS return_correlation
FROM ohlcv a
JOIN ohlcv b ON a.date = b.date AND a.ticker < b.ticker
WHERE a.date >= DATE_SUB(CURRENT_DATE(), 252)
  AND a.daily_return IS NOT NULL
  AND b.daily_return IS NOT NULL
GROUP BY a.ticker, b.ticker
HAVING COUNT(*) >= 200
"""


# ---------------------------------------------------------------------------
# Job runner class
# ---------------------------------------------------------------------------

class SparkAnalyticsJob:
    """
    Wraps SparkSQL analytics queries into runnable jobs with caching,
    logging, and output materialisation.
    """

    def __init__(self, spark: SparkSession, output_path: str = "data/output"):
        self.spark = spark
        self.output_path = output_path

    def run_sector_performance(self) -> DataFrame:
        logger.info("Running sector performance job")
        df = self.spark.sql(SECTOR_PERFORMANCE_SQL)
        self._materialise(df, "sector_performance")
        return df

    def run_top_performers(self) -> DataFrame:
        logger.info("Running top performers job")
        df = self.spark.sql(TOP_PERFORMERS_SQL)
        self._materialise(df, "top_performers")
        return df

    def run_volatility_regime(self) -> DataFrame:
        logger.info("Running volatility regime job")
        df = self.spark.sql(VOLATILITY_REGIME_SQL)
        self._materialise(df, "volatility_regime")
        return df

    def run_all(self) -> dict[str, DataFrame]:
        """Run all jobs sequentially and return results."""
        return {
            "sector_performance":  self.run_sector_performance(),
            "top_performers":      self.run_top_performers(),
            "volatility_regime":   self.run_volatility_regime(),
        }

    def _materialise(self, df: DataFrame, name: str) -> None:
        path = f"{self.output_path}/{name}"
        df.coalesce(1).write.mode("overwrite").parquet(path)
        logger.info(f"Materialised -> {path} ({df.count()} rows)")


# ---------------------------------------------------------------------------
# Utility: broadcast join for fundamentals enrichment
# ---------------------------------------------------------------------------

def enrich_with_fundamentals(
    ohlcv: DataFrame,
    fundamentals: DataFrame,
    cols: list[str] | None = None,
) -> DataFrame:
    """
    Broadcast-join fundamentals onto time-series OHLCV frame.
    Uses broadcast hint to avoid shuffle on the smaller fundamentals table.
    """
    if cols is None:
        cols = ["sector", "market_cap_category", "exchange", "country",
                "beta", "pe_ratio", "pb_ratio", "dividend_yield", "roe"]

    fund_subset = fundamentals.select(["ticker"] + cols)
    return ohlcv.join(F.broadcast(fund_subset), on="ticker", how="left")
