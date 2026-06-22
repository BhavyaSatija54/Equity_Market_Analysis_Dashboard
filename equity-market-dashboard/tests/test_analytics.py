"""
tests/test_analytics.py
-------------------------
Unit tests for portfolio metrics and scenario analysis.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.portfolio_metrics import (
    RiskMetrics,
    compute_portfolio_metrics,
    compute_risk_metrics,
    rolling_beta,
    rolling_sharpe,
    rolling_var,
)
from src.analytics.scenario_analysis import (
    PREDEFINED_SCENARIOS,
    ScenarioAnalysisEngine,
    sector_scenario_returns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_returns() -> pd.Series:
    """Constant 0.1% daily return (deterministic)."""
    return pd.Series([0.001] * 252)


@pytest.fixture
def volatile_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0004, 0.015, 252))


@pytest.fixture
def returns_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    mkt = rng.normal(0.0003, 0.012, 252)
    data = {
        f"T{i:03d}": rng.uniform(0.5, 1.5) * mkt + rng.normal(0, 0.01, 252)
        for i in range(10)
    }
    return pd.DataFrame(data)


@pytest.fixture
def betas_series(returns_matrix) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(
        {t: float(rng.uniform(0.6, 1.4)) for t in returns_matrix.columns}
    )


# ---------------------------------------------------------------------------
# compute_risk_metrics
# ---------------------------------------------------------------------------

class TestComputeRiskMetrics:

    def test_returns_risk_metrics_instance(self, volatile_returns):
        m = compute_risk_metrics(volatile_returns, "TEST")
        assert isinstance(m, RiskMetrics)

    def test_positive_flat_sharpe(self, flat_returns):
        m = compute_risk_metrics(flat_returns)
        # Constant 25% annual return → Sharpe should be positive
        assert m.sharpe_ratio > 0

    def test_annualised_vol_positive(self, volatile_returns):
        m = compute_risk_metrics(volatile_returns)
        assert m.annualised_volatility > 0

    def test_max_drawdown_non_positive(self, volatile_returns):
        m = compute_risk_metrics(volatile_returns)
        assert m.max_drawdown <= 0

    def test_var_95_less_than_var_99(self, volatile_returns):
        m = compute_risk_metrics(volatile_returns)
        # 99% VaR is more extreme (more negative)
        assert m.var_99 < m.var_95

    def test_win_rate_between_0_and_1(self, volatile_returns):
        m = compute_risk_metrics(volatile_returns)
        assert 0 <= m.win_rate <= 1

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Insufficient"):
            compute_risk_metrics(pd.Series([0.01]))

    def test_nan_handling(self):
        r = pd.Series([0.01, np.nan, 0.02, np.nan, 0.015] * 50)
        m = compute_risk_metrics(r)
        assert m.annualised_volatility > 0

    def test_ticker_passed_through(self, volatile_returns):
        m = compute_risk_metrics(volatile_returns, ticker="AAPL")
        assert m.ticker == "AAPL"


# ---------------------------------------------------------------------------
# compute_portfolio_metrics
# ---------------------------------------------------------------------------

class TestComputePortfolioMetrics:

    def test_equal_weight_returns_metrics(self, returns_matrix):
        weights = {t: 1.0 for t in returns_matrix.columns}
        m = compute_portfolio_metrics(returns_matrix, weights)
        assert m.annualised_volatility > 0
        assert m.sharpe_ratio is not None

    def test_weights_normalised(self, returns_matrix):
        weights = {t: 100.0 for t in returns_matrix.columns}   # intentionally not summing to 1
        m = compute_portfolio_metrics(returns_matrix, weights)
        assert abs(sum(m.weights.values()) - 1.0) < 1e-6

    def test_single_asset_portfolio(self, returns_matrix):
        single = {"T000": 1.0}
        m = compute_portfolio_metrics(returns_matrix, single)
        assert m.annualised_volatility > 0

    def test_with_benchmark(self, returns_matrix):
        weights = {t: 1.0 for t in returns_matrix.columns}
        benchmark = returns_matrix["T000"]
        m = compute_portfolio_metrics(returns_matrix, weights, benchmark_returns=benchmark)
        assert m.beta is not None
        assert m.alpha is not None
        assert m.treynor_ratio is not None

    def test_missing_tickers_ignored(self, returns_matrix):
        weights = {"T000": 0.5, "GHOST": 0.5}   # GHOST not in matrix
        m = compute_portfolio_metrics(returns_matrix, weights)
        assert "T000" in m.weights
        assert "GHOST" not in m.weights

    def test_max_drawdown_non_positive(self, returns_matrix):
        weights = {t: 1.0 for t in returns_matrix.columns}
        m = compute_portfolio_metrics(returns_matrix, weights)
        assert m.max_drawdown <= 0


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

class TestRollingMetrics:

    def test_rolling_sharpe_length(self, volatile_returns):
        rs = rolling_sharpe(volatile_returns, window=60)
        assert len(rs) == len(volatile_returns)

    def test_rolling_sharpe_nan_at_start(self, volatile_returns):
        rs = rolling_sharpe(volatile_returns, window=60)
        assert rs.iloc[:59].isna().all()

    def test_rolling_beta_finite_after_window(self, volatile_returns):
        bench = volatile_returns * 0.8 + pd.Series(np.random.normal(0, 0.01, len(volatile_returns)))
        rb = rolling_beta(volatile_returns, bench, window=60)
        assert rb.iloc[59:].notna().any()

    def test_rolling_var_all_negative(self, volatile_returns):
        rv = rolling_var(volatile_returns, window=30)
        valid = rv.dropna()
        assert (valid <= 0).all()


# ---------------------------------------------------------------------------
# ScenarioAnalysisEngine
# ---------------------------------------------------------------------------

class TestScenarioAnalysisEngine:

    def test_run_all_scenarios_returns_three(self, returns_matrix, betas_series):
        engine = ScenarioAnalysisEngine()
        results = engine.run_all_scenarios(returns_matrix, betas_series)
        assert set(results.keys()) == {"bull", "base", "bear"}

    def test_bull_return_greater_than_bear(self, returns_matrix, betas_series):
        engine = ScenarioAnalysisEngine()
        results = engine.run_all_scenarios(returns_matrix, betas_series)
        assert results["bull"].portfolio_return > results["bear"].portfolio_return

    def test_bear_vol_greater_than_bull(self, returns_matrix, betas_series):
        engine = ScenarioAnalysisEngine()
        results = engine.run_all_scenarios(returns_matrix, betas_series)
        assert results["bear"].portfolio_vol > results["bull"].portfolio_vol

    def test_all_tickers_have_scenario_return(self, returns_matrix, betas_series):
        engine = ScenarioAnalysisEngine()
        result = engine.apply_scenario(
            returns_matrix, betas_series, PREDEFINED_SCENARIOS["base"]
        )
        for t in returns_matrix.columns:
            if t in betas_series.index:
                assert t in result.ticker_returns

    def test_custom_scenario_subset(self, returns_matrix, betas_series):
        engine = ScenarioAnalysisEngine()
        results = engine.run_all_scenarios(
            returns_matrix, betas_series, scenarios=["bull", "stagflation"]
        )
        assert "bull" in results
        assert "stagflation" in results
        assert "base" not in results

    def test_upside_downside_capture_dataframe(self, returns_matrix, betas_series):
        engine = ScenarioAnalysisEngine()
        df = engine.upside_downside_capture(returns_matrix, betas_series)
        assert "upside_capture" in df.columns
        assert "downside_capture" in df.columns
        assert "capture_ratio" in df.columns
        assert len(df) > 0

    def test_equal_weight_vs_custom_weight(self, returns_matrix, betas_series):
        engine = ScenarioAnalysisEngine()
        # Heavy weight on first ticker
        weights = {"T000": 0.9, **{t: 0.1 / (len(returns_matrix.columns) - 1)
                                   for t in returns_matrix.columns if t != "T000"}}
        r_custom = engine.apply_scenario(
            returns_matrix, betas_series, PREDEFINED_SCENARIOS["base"], weights=weights
        )
        r_equal  = engine.apply_scenario(
            returns_matrix, betas_series, PREDEFINED_SCENARIOS["base"], weights=None
        )
        # Results should differ
        assert r_custom.portfolio_return != r_equal.portfolio_return


# ---------------------------------------------------------------------------
# sector_scenario_returns
# ---------------------------------------------------------------------------

class TestSectorScenarioReturns:

    def test_returns_all_sectors(self):
        sr = sector_scenario_returns("bull")
        assert len(sr) == 11

    def test_bull_returns_generally_positive(self):
        sr = sector_scenario_returns("bull")
        assert sum(1 for v in sr.values() if v > 0) >= 8

    def test_bear_returns_mostly_negative(self):
        # Cyclical sectors should be negative; defensive sectors (Utilities, Staples)
        # may post positive returns in a bear market due to flight-to-safety
        sr = sector_scenario_returns("bear")
        negative_count = sum(1 for v in sr.values() if v < 0)
        assert negative_count >= 7, f"Expected >= 7 negative sectors in bear, got {negative_count}"

    def test_unknown_scenario_raises(self):
        with pytest.raises(KeyError):
            sector_scenario_returns("apocalypse")

    @pytest.mark.parametrize("scenario", ["bull", "base", "bear", "stagflation", "rate_shock"])
    def test_all_predefined_scenarios_work(self, scenario):
        sr = sector_scenario_returns(scenario)
        assert isinstance(sr, dict)
        assert len(sr) > 0
