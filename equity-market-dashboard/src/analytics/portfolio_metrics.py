"""
src/analytics/portfolio_metrics.py
------------------------------------
Pure-Python (NumPy/Pandas) analytics for risk metrics and portfolio stats.
Used by the FastAPI layer; independent of Spark for low-latency API responses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

TRADING_DAYS = 252
DEFAULT_RFR  = 0.05   # annualised risk-free rate


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    ticker: str
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float           # 1-day 95% parametric VaR (as % of value)
    cvar_95: float          # Expected shortfall
    var_99: float
    win_rate: float
    best_day: float
    worst_day: float
    skewness: float
    kurtosis: float


@dataclass
class PortfolioMetrics:
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
    weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Single-equity metrics
# ---------------------------------------------------------------------------

def compute_risk_metrics(
    returns: pd.Series,
    ticker: str = "UNKNOWN",
    rfr: float = DEFAULT_RFR,
) -> RiskMetrics:
    """
    Compute full risk metrics from a daily return series.
    Handles NaN gracefully.
    """
    r = returns.dropna()
    n = len(r)

    if n < 2:
        raise ValueError(f"Insufficient return data for {ticker}: {n} observations")

    total_ret    = float((1 + r).prod() - 1)
    years        = n / TRADING_DAYS
    ann_ret      = float((1 + total_ret) ** (1 / max(years, 1)) - 1)
    ann_vol      = float(r.std() * np.sqrt(TRADING_DAYS))

    # Sharpe
    excess       = ann_ret - rfr
    sharpe       = excess / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation)
    downside     = r[r < 0]
    down_std     = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else ann_vol
    sortino      = excess / down_std if down_std > 0 else 0.0

    # Max drawdown
    cum          = (1 + r).cumprod()
    roll_max     = cum.cummax()
    drawdown     = (cum - roll_max) / roll_max
    max_dd       = float(drawdown.min())

    # Calmar
    calmar       = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    # VaR / CVaR (parametric — normal assumption)
    mu_d, sig_d  = float(r.mean()), float(r.std())
    var_95       = float(mu_d - 1.6449 * sig_d)   # 1-day 95%
    var_99       = float(mu_d - 2.3263 * sig_d)
    cvar_95      = float(r[r <= var_95].mean()) if len(r[r <= var_95]) > 0 else var_95

    # Distributional stats
    skew         = float(r.skew())
    kurt         = float(r.kurtosis())
    win_rate     = float((r > 0).mean())
    best_day     = float(r.max())
    worst_day    = float(r.min())

    return RiskMetrics(
        ticker=ticker,
        total_return=round(total_ret, 6),
        annualised_return=round(ann_ret, 6),
        annualised_volatility=round(ann_vol, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 6),
        calmar_ratio=round(calmar, 4),
        var_95=round(var_95, 6),
        cvar_95=round(cvar_95, 6),
        var_99=round(var_99, 6),
        win_rate=round(win_rate, 4),
        best_day=round(best_day, 6),
        worst_day=round(worst_day, 6),
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
    )


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------

def compute_portfolio_metrics(
    returns_matrix: pd.DataFrame,
    weights: dict[str, float],
    benchmark_returns: Optional[pd.Series] = None,
    rfr: float = DEFAULT_RFR,
) -> PortfolioMetrics:
    """
    Portfolio-level analytics given a returns matrix (columns = tickers)
    and a weight dict.

    Args:
        returns_matrix : DataFrame with tickers as columns, dates as index
        weights        : {ticker: weight}, should sum to ~1.0
        benchmark_returns : optional benchmark daily returns
        rfr            : annualised risk-free rate
    """
    # Align weights to columns
    tickers = [t for t in weights if t in returns_matrix.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()                        # normalise
    R = returns_matrix[tickers].dropna()

    port_returns = R @ w                   # weighted daily return series

    total_ret = float((1 + port_returns).prod() - 1)
    years     = len(port_returns) / TRADING_DAYS
    ann_ret   = float((1 + total_ret) ** (1 / max(years, 1)) - 1)
    ann_vol   = float(port_returns.std() * np.sqrt(TRADING_DAYS))
    excess    = ann_ret - rfr

    sharpe    = excess / ann_vol if ann_vol > 0 else 0.0

    downside  = port_returns[port_returns < 0]
    down_std  = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else ann_vol
    sortino   = excess / down_std if down_std > 0 else 0.0

    cum       = (1 + port_returns).cumprod()
    roll_max  = cum.cummax()
    max_dd    = float(((cum - roll_max) / roll_max).min())
    calmar    = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    mu_d, sig_d = float(port_returns.mean()), float(port_returns.std())
    var_95   = float(mu_d - 1.6449 * sig_d)
    var_99   = float(mu_d - 2.3263 * sig_d)
    cvar_95  = float(port_returns[port_returns <= var_95].mean()) \
               if len(port_returns[port_returns <= var_95]) > 0 else var_95

    # Market beta and alpha
    beta = alpha = treynor = ir = None
    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(port_returns.index).dropna()
        p  = port_returns.reindex(bm.index)
        cov = np.cov(p, bm)
        if cov[1, 1] > 0:
            beta    = float(cov[0, 1] / cov[1, 1])
            bm_ann  = float(bm.mean() * TRADING_DAYS)
            alpha   = ann_ret - (rfr + beta * (bm_ann - rfr))
            treynor = excess / beta if beta != 0 else None
            te      = float((p - bm).std() * np.sqrt(TRADING_DAYS))
            ir      = (ann_ret - bm_ann) / te if te > 0 else None

    return PortfolioMetrics(
        total_return=round(total_ret, 6),
        annualised_return=round(ann_ret, 6),
        annualised_volatility=round(ann_vol, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 6),
        calmar_ratio=round(calmar, 4),
        var_95=round(var_95, 6),
        cvar_95=round(cvar_95, 6),
        var_99=round(var_99, 6),
        beta=round(beta, 4) if beta is not None else None,
        alpha=round(alpha, 4) if alpha is not None else None,
        treynor_ratio=round(treynor, 4) if treynor is not None else None,
        information_ratio=round(ir, 4) if ir is not None else None,
        weights=dict(zip(tickers, w.tolist())),
    )


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def rolling_sharpe(returns: pd.Series, window: int = 252, rfr: float = DEFAULT_RFR) -> pd.Series:
    mu_ann  = returns.rolling(window).mean() * TRADING_DAYS
    vol_ann = returns.rolling(window).std()  * np.sqrt(TRADING_DAYS)
    return (mu_ann - rfr) / vol_ann.replace(0, np.nan)


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
    cov = returns.rolling(window).cov(benchmark)
    var = benchmark.rolling(window).var()
    return cov / var.replace(0, np.nan)


def rolling_var(returns: pd.Series, window: int = 60, confidence: float = 0.95) -> pd.Series:
    z = 1.6449 if confidence == 0.95 else 2.3263
    mu  = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    return mu - z * sig
