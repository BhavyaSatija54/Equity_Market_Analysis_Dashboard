"""
api/models/schemas.py
-----------------------
Pydantic v2 request and response schemas for all API endpoints.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Equity schemas
# ---------------------------------------------------------------------------

class EquityBasic(BaseModel):
    ticker: str
    sector: str
    market_cap_category: str
    exchange: str
    country: str
    beta: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    dividend_yield: Optional[float]
    roe: Optional[float]
    debt_to_equity: Optional[float]

    # Latest price snapshot
    close: Optional[float] = None
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_1y: Optional[float] = None
    vol_20d: Optional[float] = None
    rsi_14: Optional[float] = None
    momentum_zscore: Optional[float] = None


class EquityDetail(EquityBasic):
    """Extended equity profile with historical prices."""
    history: list[dict] = Field(default_factory=list)
    risk: Optional["RiskMetricsSchema"] = None


class EquityScreenerRequest(BaseModel):
    sectors: Optional[list[str]] = None
    market_caps: Optional[list[str]] = None
    exchanges: Optional[list[str]] = None
    min_beta: Optional[float] = None
    max_beta: Optional[float] = None
    min_pe: Optional[float] = None
    max_pe: Optional[float] = None
    min_return_1y: Optional[float] = None
    max_return_1y: Optional[float] = None
    sort_by: str = "return_1y"
    sort_desc: bool = True
    limit: int = Field(default=100, le=500)
    offset: int = 0


class EquityScreenerResponse(BaseModel):
    total: int
    equities: list[EquityBasic]


# ---------------------------------------------------------------------------
# Risk metric schemas
# ---------------------------------------------------------------------------

class RiskMetricsSchema(BaseModel):
    ticker: str
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    var_99: float
    win_rate: float
    best_day: float
    worst_day: float
    skewness: float
    kurtosis: float


class PortfolioRiskRequest(BaseModel):
    weights: dict[str, float] = Field(
        description="Ticker -> weight mapping (need not sum to 1; will be normalised)"
    )
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @field_validator("weights")
    @classmethod
    def at_least_one_ticker(cls, v):
        if len(v) == 0:
            raise ValueError("At least one ticker required")
        return v


class PortfolioRiskResponse(BaseModel):
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    var_99: float
    beta: Optional[float]
    alpha: Optional[float]
    treynor_ratio: Optional[float]
    information_ratio: Optional[float]
    weights: dict[str, float]
    constituent_metrics: list[RiskMetricsSchema]


# ---------------------------------------------------------------------------
# Scenario schemas
# ---------------------------------------------------------------------------

class ScenarioResultSchema(BaseModel):
    scenario_name: str
    portfolio_return: float
    portfolio_vol: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    ticker_returns: dict[str, float] = Field(default_factory=dict)


class ScenarioRequest(BaseModel):
    tickers: Optional[list[str]] = None
    weights: Optional[dict[str, float]] = None
    scenarios: list[str] = Field(
        default=["bull", "base", "bear"],
        description="Scenario keys: bull | base | bear | stagflation | rate_shock"
    )


class ScenarioResponse(BaseModel):
    scenarios: dict[str, ScenarioResultSchema]
    sector_scenario_returns: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Market summary schemas
# ---------------------------------------------------------------------------

class SectorSummary(BaseModel):
    sector: str
    n_equities: int
    annualised_avg_return: float
    annualised_volatility: float
    sharpe_ratio: float
    avg_beta: float
    avg_pe: Optional[float]
    win_rate: float


class MarketSummaryResponse(BaseModel):
    as_of_date: date
    total_equities: int
    market_avg_return_1y: float
    market_avg_vol: float
    market_breadth_pct: float           # % stocks above 50-day SMA
    sectors: list[SectorSummary]
    top_performers: list[EquityBasic]
    bottom_performers: list[EquityBasic]
    vol_regime: str                     # High | Normal | Low


# ---------------------------------------------------------------------------
# Reporting schemas
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    title: str = "Equity Market Analysis Report"
    tickers: Optional[list[str]] = None
    format: str = Field(default="html", pattern="^(html|pdf|excel)$")
    sections: list[str] = Field(
        default=["market_summary", "top_performers", "risk_metrics",
                 "scenario_analysis", "sector_breakdown"]
    )


class ReportResponse(BaseModel):
    report_id: str
    status: str
    download_url: Optional[str] = None
    message: str
