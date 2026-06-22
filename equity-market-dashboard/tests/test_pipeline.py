"""
tests/test_pipeline.py
------------------------
Unit and integration tests for the PySpark ingestion and transformation pipeline.
Uses a local SparkSession so no cluster is required.
"""

import os
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

# Guard: skip if PySpark unavailable (CI without Java)
pytest.importorskip("pyspark", reason="PySpark not available")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from src.data_pipeline.ingestion import (
    FUNDAMENTALS_SCHEMA,
    OHLCV_SCHEMA,
    EquityDataIngestion,
    IngestionConfig,
    create_spark_session,
)
from src.data_pipeline.transformation import (
    add_cumulative_metrics,
    add_momentum_indicators,
    add_moving_averages,
    add_returns,
    add_volatility_features,
    build_feature_set,
)
from src.data_pipeline.spark_jobs import enrich_with_fundamentals


# ---------------------------------------------------------------------------
# Session fixture (session-scoped — one Spark instance per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .appName("test_pipeline")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_pdf(n_tickers: int = 5, n_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end = date.today()
    dates = [end - timedelta(days=i) for i in range(n_days - 1, -1, -1)]
    dates = [d for d in dates if d.weekday() < 5][:n_days]

    rows = []
    sectors = ["Technology", "Financials", "Healthcare", "Industrials", "Energy"]
    for j in range(n_tickers):
        ticker = f"T{j:03d}"
        p = 100 + rng.uniform(0, 100)
        for d in dates:
            ret = rng.normal(0.0003, 0.012)
            p   = p * (1 + ret)
            rng2 = rng.uniform(0.005, 0.02)
            rows.append({
                "ticker":  ticker,
                "date":    d,
                "open":    round(p * (1 - rng2 * 0.5), 2),
                "high":    round(p * (1 + rng2), 2),
                "low":     round(p * (1 - rng2), 2),
                "close":   round(p, 2),
                "volume":  int(rng.integers(100_000, 5_000_000)),
                "sector":  sectors[j % len(sectors)],
            })
    return pd.DataFrame(rows)


def _make_fund_pdf(n_tickers: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    tickers = [f"T{j:03d}" for j in range(n_tickers)]
    sectors = ["Technology", "Financials", "Healthcare", "Industrials", "Energy"]
    return pd.DataFrame({
        "ticker":              tickers,
        "sector":              [sectors[j % len(sectors)] for j in range(n_tickers)],
        "market_cap_category": ["Large Cap"] * n_tickers,
        "exchange":            ["NYSE"] * n_tickers,
        "country":             ["US"] * n_tickers,
        "beta":                list(rng.uniform(0.6, 1.5, n_tickers)),
        "pe_ratio":            list(rng.uniform(10, 40, n_tickers)),
        "pb_ratio":            list(rng.uniform(1, 5, n_tickers)),
        "dividend_yield":      list(rng.uniform(0, 0.05, n_tickers)),
        "roe":                 list(rng.uniform(0.05, 0.30, n_tickers)),
        "debt_to_equity":      list(rng.uniform(0, 1.5, n_tickers)),
    })


@pytest.fixture(scope="module")
def ohlcv_df(spark):
    pdf = _make_ohlcv_pdf(n_tickers=5, n_days=120)
    return spark.createDataFrame(pdf, schema=OHLCV_SCHEMA)


@pytest.fixture(scope="module")
def fund_df(spark):
    pdf = _make_fund_pdf(n_tickers=5)
    return spark.createDataFrame(pdf, schema=FUNDAMENTALS_SCHEMA)


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestEquityDataIngestion:

    def test_create_spark_session(self, spark):
        assert spark is not None
        assert spark.version is not None

    def test_ingestion_run_returns_dataframes(self, spark, tmp_path):
        pdf_ohlcv = _make_ohlcv_pdf(n_tickers=3, n_days=300)
        pdf_fund  = _make_fund_pdf(n_tickers=3)

        ohlcv_path = str(tmp_path / "ohlcv.parquet")
        fund_path  = str(tmp_path / "fund.parquet")
        out_path   = str(tmp_path / "out")

        pdf_ohlcv.to_parquet(ohlcv_path, index=False)
        pdf_fund.to_parquet(fund_path, index=False)

        cfg = IngestionConfig(
            ohlcv_path=ohlcv_path,
            fundamentals_path=fund_path,
            output_path=out_path,
            min_trading_days=50,
        )
        ingest = EquityDataIngestion(spark, cfg)
        result = ingest.run()

        assert "ohlcv" in result
        assert "fundamentals" in result
        assert result["ohlcv"].count() > 0

    def test_quality_check_null_close_raises(self, spark, tmp_path):
        pdf = _make_ohlcv_pdf(n_tickers=2, n_days=300)
        # Force 50% null closes to trip threshold
        null_idx = pdf.sample(frac=0.5, random_state=0).index
        pdf.loc[null_idx, "close"] = None

        ohlcv_path = str(tmp_path / "ohlcv_null.parquet")
        fund_path  = str(tmp_path / "fund_null.parquet")
        out_path   = str(tmp_path / "out_null")

        pdf.to_parquet(ohlcv_path, index=False)
        _make_fund_pdf(2).to_parquet(fund_path, index=False)

        cfg = IngestionConfig(
            ohlcv_path=ohlcv_path,
            fundamentals_path=fund_path,
            output_path=out_path,
            min_trading_days=5,
            max_null_pct=0.05,
        )
        ingest = EquityDataIngestion(spark, cfg)
        with pytest.raises(ValueError, match="Null close rate"):
            ingest.run()

    def test_min_trading_days_filter(self, spark, tmp_path):
        # One ticker with 200 days, one with 30 days
        pdf1 = _make_ohlcv_pdf(n_tickers=1, n_days=200)
        pdf2 = _make_ohlcv_pdf(n_tickers=1, n_days=30)
        pdf2["ticker"] = "SHORT"
        pdf  = pd.concat([pdf1, pdf2])

        ohlcv_path = str(tmp_path / "ohlcv_min.parquet")
        fund_path  = str(tmp_path / "fund_min.parquet")
        out_path   = str(tmp_path / "out_min")

        pdf.to_parquet(ohlcv_path, index=False)
        _make_fund_pdf(1).to_parquet(fund_path, index=False)

        cfg = IngestionConfig(
            ohlcv_path=ohlcv_path,
            fundamentals_path=fund_path,
            output_path=out_path,
            min_trading_days=100,
        )
        ingest = EquityDataIngestion(spark, cfg)
        result = ingest.run()

        tickers = {r["ticker"] for r in result["ohlcv"].select("ticker").distinct().collect()}
        assert "SHORT" not in tickers


# ---------------------------------------------------------------------------
# Transformation tests
# ---------------------------------------------------------------------------

class TestAddReturns:

    def test_adds_return_columns(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        assert "daily_return" in out.columns
        assert "log_return" in out.columns

    def test_first_row_return_null(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        nulls = out.filter(F.col("daily_return").isNull()).count()
        # Should have exactly n_tickers null rows (first day per ticker)
        assert nulls == ohlcv_df.select("ticker").distinct().count()

    def test_returns_finite(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        inf_count = out.filter(F.col("daily_return") == float("inf")).count()
        assert inf_count == 0


class TestAddMovingAverages:

    def test_sma_columns_present(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        out = add_moving_averages(out, windows=[20, 50])
        assert "sma_20" in out.columns
        assert "sma_50" in out.columns

    def test_sma_null_before_window(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        out = add_moving_averages(out, windows=[20])
        # For each ticker, first 19 rows of sma_20 should be null
        first_ticker = out.filter(F.col("ticker") == "T000").orderBy("date")
        head_nulls = first_ticker.limit(19).filter(F.col("sma_20").isNull()).count()
        assert head_nulls == 19

    def test_bollinger_bands_present(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        out = add_moving_averages(out)
        assert "bb_upper" in out.columns
        assert "bb_lower" in out.columns

    def test_upper_gt_lower(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        out = add_moving_averages(out)
        violations = out.filter(
            F.col("bb_upper").isNotNull() &
            (F.col("bb_upper") < F.col("bb_lower"))
        ).count()
        assert violations == 0


class TestAddMomentum:

    def test_rsi_range(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        out = add_momentum_indicators(out)
        rsi = out.filter(F.col("rsi_14").isNotNull())
        below_0  = rsi.filter(F.col("rsi_14") < 0).count()
        above_100= rsi.filter(F.col("rsi_14") > 100).count()
        assert below_0  == 0
        assert above_100 == 0


class TestAddVolatility:

    def test_vol_columns_present(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        out = add_volatility_features(out)
        assert "vol_20d" in out.columns
        assert "vol_60d" in out.columns
        assert "atr_14"  in out.columns

    def test_vol_non_negative(self, ohlcv_df):
        out = add_returns(ohlcv_df)
        out = add_volatility_features(out)
        neg = out.filter(
            F.col("vol_20d").isNotNull() & (F.col("vol_20d") < 0)
        ).count()
        assert neg == 0


class TestBuildFeatureSet:

    def test_feature_set_has_expected_columns(self, ohlcv_df):
        out = build_feature_set(ohlcv_df)
        expected = ["daily_return", "log_return", "sma_20", "sma_50",
                    "rsi_14", "vol_20d", "drawdown"]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_intermediate_columns_leaked(self, ohlcv_df):
        out = build_feature_set(ohlcv_df)
        leaked = ["gain", "loss", "avg_gain", "avg_loss", "rs",
                  "prev_close", "signed_volume"]
        for col in leaked:
            assert col not in out.columns, f"Intermediate column leaked: {col}"


# ---------------------------------------------------------------------------
# Broadcast join
# ---------------------------------------------------------------------------

class TestEnrichWithFundamentals:

    def test_columns_added(self, ohlcv_df, fund_df):
        out = add_returns(ohlcv_df)
        enriched = enrich_with_fundamentals(out, fund_df)
        assert "sector" in enriched.columns
        assert "beta"   in enriched.columns

    def test_row_count_unchanged(self, ohlcv_df, fund_df):
        out = add_returns(ohlcv_df)
        enriched = enrich_with_fundamentals(out, fund_df)
        assert enriched.count() == out.count()
