"""
src/analytics/scenario_analysis.py
-------------------------------------
Bull / Base / Bear scenario stress testing for equity portfolios.
Uses historical simulation and parametric shock application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class MarketScenario:
    name: str                           # e.g. "Bull", "Base", "Bear"
    market_return_shock: float          # annual shock to market return
    volatility_multiplier: float        # scale factor for vol (>1 = more volatile)
    correlation_shift: float            # additive shift to inter-stock correlation
    description: str = ""


PREDEFINED_SCENARIOS: dict[str, MarketScenario] = {
    "bull": MarketScenario(
        name="Bull",
        market_return_shock=0.25,
        volatility_multiplier=0.75,
        correlation_shift=-0.10,
        description="Strong risk-on rally; volatility suppressed, correlations fall",
    ),
    "base": MarketScenario(
        name="Base",
        market_return_shock=0.08,
        volatility_multiplier=1.00,
        correlation_shift=0.00,
        description="Trend-growth, no major macro disruptions",
    ),
    "bear": MarketScenario(
        name="Bear",
        market_return_shock=-0.30,
        volatility_multiplier=1.80,
        correlation_shift=0.35,
        description="Severe risk-off: correlations spike, volatility surges",
    ),
    "stagflation": MarketScenario(
        name="Stagflation",
        market_return_shock=-0.15,
        volatility_multiplier=1.40,
        correlation_shift=0.20,
        description="High inflation + low growth; value rotation away from growth",
    ),
    "rate_shock": MarketScenario(
        name="Rate Shock",
        market_return_shock=-0.10,
        volatility_multiplier=1.30,
        correlation_shift=0.15,
        description="+200bps rapid rate rise; duration-sensitive sectors hit hardest",
    ),
}


# ---------------------------------------------------------------------------
# Scenario output containers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_name: str
    portfolio_return: float
    portfolio_vol: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sector_returns: dict[str, float] = field(default_factory=dict)
    ticker_returns: dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioComparison:
    ticker: str
    bull_return: float
    base_return: float
    bear_return: float
    bull_var95: float
    bear_var95: float
    upside_capture: float    # bull return / market bull return
    downside_capture: float  # bear return / market bear return


# ---------------------------------------------------------------------------
# Core scenario engine
# ---------------------------------------------------------------------------

class ScenarioAnalysisEngine:
    """
    Applies macro scenarios to a returns matrix and computes shocked metrics.
    """

    def __init__(self, rfr: float = 0.05):
        self.rfr = rfr

    def apply_scenario(
        self,
        returns: pd.DataFrame,               # (dates x tickers)
        betas: pd.Series,                    # {ticker: beta}
        scenario: MarketScenario,
        weights: Optional[dict[str, float]] = None,
    ) -> ScenarioResult:
        """
        Simulate 1-year returns under a given scenario.

        Method:
        1. Generate synthetic market returns with shocked drift + scaled vol
        2. Derive equity returns via CAPM: r_i = rf + beta_i*(r_m - rf) + alpha_i + epsilon_i
        3. Compute portfolio-level metrics from scenario returns
        """
        rng = np.random.default_rng(42)
        tickers = [t for t in returns.columns if t in betas.index]

        # Historical idiosyncratic vol
        hist_vols = returns[tickers].std() * np.sqrt(TRADING_DAYS)
        idio_vols = hist_vols * scenario.volatility_multiplier

        # Shocked market return (daily)
        mkt_ann   = scenario.market_return_shock
        mkt_vol   = 0.18 * scenario.volatility_multiplier
        mkt_drift = (mkt_ann / TRADING_DAYS)
        mkt_daily = rng.normal(mkt_drift, mkt_vol / np.sqrt(TRADING_DAYS), TRADING_DAYS)

        # Per-ticker shocked returns via CAPM
        ticker_annual_returns = {}
        ticker_daily_returns  = {}

        for t in tickers:
            b         = float(betas.get(t, 1.0))
            idio_vol  = float(idio_vols.get(t, 0.20))
            eps       = rng.normal(0, idio_vol / np.sqrt(TRADING_DAYS), TRADING_DAYS)
            r_daily   = self.rfr / TRADING_DAYS + b * (mkt_daily - self.rfr / TRADING_DAYS) + eps
            ticker_daily_returns[t]  = r_daily
            ticker_annual_returns[t] = float((1 + pd.Series(r_daily)).prod() - 1)

        # Portfolio returns
        if weights is None:
            w = np.ones(len(tickers)) / len(tickers)
        else:
            w_raw = np.array([weights.get(t, 0) for t in tickers], dtype=float)
            w     = w_raw / w_raw.sum() if w_raw.sum() > 0 else np.ones(len(tickers)) / len(tickers)

        daily_matrix = np.column_stack([ticker_daily_returns[t] for t in tickers])
        port_daily   = daily_matrix @ w
        port_return  = float((1 + pd.Series(port_daily)).prod() - 1)
        port_vol     = float(port_daily.std() * np.sqrt(TRADING_DAYS))

        # VaR / CVaR (historical on simulated path)
        var_95  = float(np.percentile(port_daily, 5))
        cvar_95 = float(port_daily[port_daily <= var_95].mean())

        # Max drawdown
        cum     = (1 + pd.Series(port_daily)).cumprod()
        max_dd  = float(((cum - cum.cummax()) / cum.cummax()).min())

        sharpe  = (port_return - self.rfr) / port_vol if port_vol > 0 else 0.0

        return ScenarioResult(
            scenario_name=scenario.name,
            portfolio_return=round(port_return, 6),
            portfolio_vol=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            var_95=round(var_95, 6),
            cvar_95=round(cvar_95, 6),
            max_drawdown=round(max_dd, 6),
            ticker_returns={t: round(v, 6) for t, v in ticker_annual_returns.items()},
        )

    def run_all_scenarios(
        self,
        returns: pd.DataFrame,
        betas: pd.Series,
        weights: Optional[dict[str, float]] = None,
        scenarios: Optional[list[str]] = None,
    ) -> dict[str, ScenarioResult]:
        """Run multiple scenarios and return results keyed by name."""
        if scenarios is None:
            scenarios = ["bull", "base", "bear"]
        return {
            k: self.apply_scenario(returns, betas, PREDEFINED_SCENARIOS[k], weights)
            for k in scenarios
            if k in PREDEFINED_SCENARIOS
        }

    def upside_downside_capture(
        self,
        returns: pd.DataFrame,
        betas: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute upside/downside capture for each ticker.
        Capture = ticker scenario return / market scenario return.
        """
        bull = PREDEFINED_SCENARIOS["bull"]
        bear = PREDEFINED_SCENARIOS["bear"]

        records = []
        for t in returns.columns:
            if t not in betas.index:
                continue
            b = float(betas.get(t, 1.0))
            idio_vol = float(returns[t].std() * np.sqrt(TRADING_DAYS))

            # Simplified CAPM expectation for capture calculation
            bull_mkt = bull.market_return_shock
            bear_mkt = bear.market_return_shock

            bull_ret = self.rfr + b * (bull_mkt - self.rfr)
            bear_ret = self.rfr + b * (bear_mkt - self.rfr)

            records.append({
                "ticker":            t,
                "bull_return":       round(bull_ret, 4),
                "bear_return":       round(bear_ret, 4),
                "base_return":       round(self.rfr + b * (0.08 - self.rfr), 4),
                "upside_capture":    round(bull_ret / bull_mkt if bull_mkt != 0 else 1.0, 4),
                "downside_capture":  round(bear_ret / bear_mkt if bear_mkt != 0 else 1.0, 4),
                "beta":              round(b, 3),
            })

        df = pd.DataFrame(records)
        df["capture_ratio"] = df["upside_capture"] / df["downside_capture"].abs()
        return df.sort_values("capture_ratio", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sector-level scenario utilities
# ---------------------------------------------------------------------------

SECTOR_SENSITIVITIES: dict[str, dict[str, float]] = {
    # Multiplier applied directly to the scenario market return
    # Positive = amplifies market move; <1 = dampens; can differ by scenario
    "Technology":              {"bull": 1.30, "base": 1.00, "bear": 1.40},
    "Financials":              {"bull": 1.15, "base": 1.00, "bear": 1.20},
    "Healthcare":              {"bull": 0.85, "base": 0.90, "bear": 0.70},
    "Consumer Discretionary":  {"bull": 1.20, "base": 1.00, "bear": 1.30},
    "Consumer Staples":        {"bull": 0.60, "base": 0.80, "bear": 0.45},
    "Energy":                  {"bull": 1.25, "base": 0.95, "bear": 1.10},
    "Industrials":             {"bull": 1.10, "base": 1.00, "bear": 1.05},
    "Utilities":               {"bull": 0.40, "base": 0.70, "bear": 0.35},
    "Materials":               {"bull": 1.10, "base": 0.95, "bear": 1.00},
    "Real Estate":             {"bull": 0.90, "base": 0.85, "bear": 1.15},
    "Communication Services":  {"bull": 1.05, "base": 0.95, "bear": 1.10},
}


def sector_scenario_returns(scenario_key: str) -> dict[str, float]:
    """
    Return sector-level expected returns for a given scenario.
    """
    scen = PREDEFINED_SCENARIOS[scenario_key]
    results = {}
    for sector, sens in SECTOR_SENSITIVITIES.items():
        mult = sens.get(scenario_key, 1.0)
        results[sector] = round(scen.market_return_shock * mult, 4)
    return results
