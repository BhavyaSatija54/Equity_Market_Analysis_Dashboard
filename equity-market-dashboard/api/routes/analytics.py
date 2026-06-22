"""
api/routes/analytics.py
-------------------------
Analytics endpoints: risk metrics, scenario analysis, market summary.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.models.schemas import (
    MarketSummaryResponse,
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    RiskMetricsSchema,
    ScenarioRequest,
    ScenarioResponse,
    SectorSummary,
)
from src.analytics.portfolio_metrics import compute_portfolio_metrics, compute_risk_metrics
from src.analytics.scenario_analysis import (
    ScenarioAnalysisEngine,
    sector_scenario_returns,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared: synthetic data helpers (replace with real data loader in production)
# ---------------------------------------------------------------------------

_SECTORS = [
    "Technology", "Financials", "Healthcare", "Consumer Discretionary",
    "Industrials", "Communication Services", "Consumer Staples",
    "Energy", "Materials", "Utilities", "Real Estate",
]


def _generate_returns(tickers: list[str], n_days: int = 252) -> pd.DataFrame:
    """Generate correlated return series for demo purposes."""
    rng = np.random.default_rng(sum(ord(c) for c in "".join(tickers)))
    dates = pd.bdate_range(end=date.today(), periods=n_days)
    data = {}
    mkt = rng.normal(0.0003, 0.012, n_days)
    for t in tickers:
        beta = rng.uniform(0.6, 1.5)
        idio = rng.normal(0, 0.015, n_days)
        data[t] = beta * mkt + idio
    return pd.DataFrame(data, index=dates)


def _get_betas(tickers: list[str]) -> pd.Series:
    rng = np.random.default_rng(123)
    return pd.Series(
        {t: float(rng.uniform(0.6, 1.5)) for t in tickers}
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/risk", response_model=PortfolioRiskResponse)
async def portfolio_risk(request: PortfolioRiskRequest):
    """
    Compute portfolio-level and constituent-level risk metrics.
    Accepts a weight dict; normalises weights internally.
    """
    tickers = list(request.weights.keys())
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    n_days = 504  # 2 years
    if request.start_date and request.end_date:
        n_days = (request.end_date - request.start_date).days

    try:
        returns = _generate_returns(tickers, n_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Return generation failed: {e}")

    # Constituent metrics
    constituent = []
    for t in tickers:
        try:
            m = compute_risk_metrics(returns[t], ticker=t)
            constituent.append(RiskMetricsSchema(**m.__dict__))
        except Exception:
            continue

    # Portfolio metrics
    try:
        port = compute_portfolio_metrics(returns, request.weights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio metrics failed: {e}")

    return PortfolioRiskResponse(
        **{k: v for k, v in port.__dict__.items() if k != "weights"},
        weights=port.weights,
        constituent_metrics=constituent,
    )


@router.post("/scenario", response_model=ScenarioResponse)
async def scenario_analysis(request: ScenarioRequest):
    """
    Run Bull / Base / Bear (and custom) scenario stress tests.
    Returns projected portfolio returns and per-sector impacts.
    """
    tickers = request.tickers or [f"T{i:04d}" for i in range(1, 51)]
    engine  = ScenarioAnalysisEngine()
    returns = _generate_returns(tickers)
    betas   = _get_betas(tickers)

    try:
        results = engine.run_all_scenarios(
            returns=returns,
            betas=betas,
            weights=request.weights,
            scenarios=request.scenarios,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from api.models.schemas import ScenarioResultSchema
    scenario_out = {
        k: ScenarioResultSchema(**v.__dict__)
        for k, v in results.items()
    }

    sector_scen = {
        scen_key: sector_scenario_returns(scen_key)
        for scen_key in request.scenarios
        if scen_key in ("bull", "base", "bear", "stagflation", "rate_shock")
    }

    return ScenarioResponse(
        scenarios=scenario_out,
        sector_scenario_returns=sector_scen,
    )


@router.get("/market-summary", response_model=MarketSummaryResponse)
async def market_summary():
    """
    Aggregated market overview: sector stats, breadth, top/bottom performers.
    Response is cached (60s TTL) in production via Redis.
    """
    rng = np.random.default_rng(7)

    sectors_data = []
    for sec in _SECTORS:
        ann_ret = float(rng.normal(0.08, 0.12))
        ann_vol = float(rng.uniform(0.15, 0.40))
        sectors_data.append(SectorSummary(
            sector=sec,
            n_equities=int(rng.integers(20, 80)),
            annualised_avg_return=round(ann_ret, 4),
            annualised_volatility=round(ann_vol, 4),
            sharpe_ratio=round((ann_ret - 0.05) / ann_vol, 4),
            avg_beta=round(float(rng.uniform(0.6, 1.4)), 3),
            avg_pe=round(float(rng.uniform(12, 35)), 2),
            win_rate=round(float(rng.uniform(0.45, 0.58)), 4),
        ))

    from api.models.schemas import EquityBasic

    def _mock_equity(ticker: str, sector: str) -> EquityBasic:
        return EquityBasic(
            ticker=ticker,
            sector=sector,
            market_cap_category=rng.choice(["Large Cap", "Mid Cap", "Small Cap"]),
            exchange=rng.choice(["NYSE", "NASDAQ"]),
            country="US",
            beta=round(float(rng.uniform(0.5, 2.0)), 2),
            pe_ratio=round(float(rng.uniform(8, 60)), 1),
            pb_ratio=round(float(rng.uniform(0.5, 8)), 2),
            dividend_yield=round(float(rng.uniform(0, 0.06)), 4),
            roe=round(float(rng.uniform(-0.05, 0.40)), 4),
            debt_to_equity=round(float(rng.uniform(0, 2)), 3),
            close=round(float(rng.uniform(10, 2000)), 2),
            return_1y=round(float(rng.uniform(0.15, 0.80)), 4),
            return_1d=round(float(rng.normal(0, 0.015)), 4),
            vol_20d=round(float(rng.uniform(0.15, 0.50)), 4),
            rsi_14=round(float(rng.uniform(30, 80)), 1),
        )

    top_tickers    = [f"GAIN{i}" for i in range(1, 6)]
    bottom_tickers = [f"LOSS{i}" for i in range(1, 6)]

    return MarketSummaryResponse(
        as_of_date=date.today(),
        total_equities=500,
        market_avg_return_1y=round(float(rng.normal(0.08, 0.03)), 4),
        market_avg_vol=round(float(rng.uniform(0.18, 0.28)), 4),
        market_breadth_pct=round(float(rng.uniform(0.45, 0.70)), 4),
        sectors=sectors_data,
        top_performers=[_mock_equity(t, "Technology") for t in top_tickers],
        bottom_performers=[_mock_equity(t, "Energy") for t in bottom_tickers],
        vol_regime="Normal",
    )
