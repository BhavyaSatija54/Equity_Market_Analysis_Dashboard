"""
api/routes/equities.py
------------------------
Equity screener and single-equity detail endpoints.
"""

from __future__ import annotations

import numpy as np
from datetime import date
from fastapi import APIRouter, HTTPException, Query

from api.models.schemas import (
    EquityBasic,
    EquityDetail,
    EquityScreenerRequest,
    EquityScreenerResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory synthetic equity universe (replace with DB / Parquet in production)
# ---------------------------------------------------------------------------

_SECTORS = [
    "Technology", "Financials", "Healthcare", "Consumer Discretionary",
    "Industrials", "Communication Services", "Consumer Staples",
    "Energy", "Materials", "Utilities", "Real Estate",
]
_SECTOR_WEIGHTS = [0.25, 0.18, 0.14, 0.11, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.01]
_MARKET_CAPS    = ["Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
_EXCHANGES      = ["NYSE", "NASDAQ", "LSE", "NSE"]
_ALPHA          = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _build_universe(n: int = 500) -> list[EquityBasic]:
    rng = np.random.default_rng(42)

    tickers: list[str] = []
    seen: set[str] = set()
    while len(tickers) < n:
        length = rng.integers(3, 5)
        t = "".join(rng.choice(_ALPHA, length))
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    sectors = rng.choice(_SECTORS, size=n, p=_SECTOR_WEIGHTS)
    equities = []

    for i, (ticker, sector) in enumerate(zip(tickers, sectors)):
        rng_i = np.random.default_rng(i)
        # Sector-tuned parameters
        sec_idx     = _SECTORS.index(sector)
        beta_mu     = [1.15, 1.05, 0.85, 1.10, 1.00, 0.95, 0.65, 1.20, 1.10, 0.55, 0.80][sec_idx]
        beta        = float(np.clip(rng_i.normal(beta_mu, 0.20), 0.1, 2.5))
        ann_ret     = float(rng_i.normal(0.08 + 0.05 * (beta - 1), 0.15))
        ann_vol     = float(rng_i.uniform(0.15 + 0.05 * beta, 0.45))
        close       = float(rng_i.uniform(5, 2500))

        equities.append(EquityBasic(
            ticker=ticker,
            sector=sector,
            market_cap_category=str(rng_i.choice(_MARKET_CAPS, p=[0.40, 0.30, 0.20, 0.10])),
            exchange=str(rng_i.choice(_EXCHANGES, p=[0.40, 0.35, 0.15, 0.10])),
            country="US" if rng_i.random() < 0.75 else ("UK" if rng_i.random() < 0.5 else "IN"),
            beta=round(beta, 3),
            pe_ratio=round(float(np.clip(rng_i.normal(20, 8), 3, 150)), 2) if sector != "Utilities" else round(float(rng_i.normal(17, 3)), 2),
            pb_ratio=round(float(np.clip(rng_i.normal(2.5, 1.0), 0.3, 12)), 2),
            dividend_yield=round(max(float(rng_i.normal(0.02, 0.015)), 0.0), 4),
            roe=round(float(rng_i.normal(0.14, 0.07)), 4),
            debt_to_equity=round(max(float(rng_i.normal(0.6, 0.4)), 0.0), 3),
            close=round(close, 2),
            return_1d=round(float(rng_i.normal(0, 0.015)), 4),
            return_5d=round(float(rng_i.normal(ann_ret / 52, ann_vol / np.sqrt(52))), 4),
            return_1m=round(float(rng_i.normal(ann_ret / 12, ann_vol / np.sqrt(12))), 4),
            return_3m=round(float(rng_i.normal(ann_ret / 4, ann_vol / 2)), 4),
            return_1y=round(float(rng_i.normal(ann_ret, ann_vol)), 4),
            vol_20d=round(float(rng_i.uniform(0.12, 0.55)), 4),
            rsi_14=round(float(np.clip(rng_i.normal(55, 15), 5, 95)), 1),
            momentum_zscore=round(float(rng_i.normal(0, 1)), 3),
        ))

    return equities


# Build once at startup
_UNIVERSE: list[EquityBasic] = _build_universe(500)
_UNIVERSE_MAP: dict[str, EquityBasic] = {e.ticker: e for e in _UNIVERSE}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=EquityScreenerResponse)
async def screener(
    sectors: list[str] | None = Query(default=None),
    market_caps: list[str] | None = Query(default=None),
    min_beta: float | None = Query(default=None),
    max_beta: float | None = Query(default=None),
    min_pe: float | None = Query(default=None),
    max_pe: float | None = Query(default=None),
    min_return_1y: float | None = Query(default=None),
    sort_by: str = Query(default="return_1y"),
    sort_desc: bool = Query(default=True),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0),
):
    """
    Equity screener with multi-axis filtering.
    Supports sector, market cap, beta, PE, return filters.
    """
    results = _UNIVERSE

    if sectors:
        results = [e for e in results if e.sector in sectors]
    if market_caps:
        results = [e for e in results if e.market_cap_category in market_caps]
    if min_beta is not None:
        results = [e for e in results if e.beta is not None and e.beta >= min_beta]
    if max_beta is not None:
        results = [e for e in results if e.beta is not None and e.beta <= max_beta]
    if min_pe is not None:
        results = [e for e in results if e.pe_ratio is not None and e.pe_ratio >= min_pe]
    if max_pe is not None:
        results = [e for e in results if e.pe_ratio is not None and e.pe_ratio <= max_pe]
    if min_return_1y is not None:
        results = [e for e in results if e.return_1y is not None and e.return_1y >= min_return_1y]

    # Sort
    def sort_key(e: EquityBasic):
        v = getattr(e, sort_by, None)
        return (v is not None, v or 0)

    results = sorted(results, key=sort_key, reverse=sort_desc)

    total = len(results)
    paginated = results[offset: offset + limit]

    return EquityScreenerResponse(total=total, equities=paginated)


@router.get("/sectors", response_model=list[str])
async def list_sectors():
    """Return all available sectors."""
    return sorted({e.sector for e in _UNIVERSE})


@router.get("/{ticker}", response_model=EquityDetail)
async def equity_detail(ticker: str):
    """
    Return full equity profile for a given ticker,
    including historical price data (last 252 trading days).
    """
    ticker = ticker.upper()
    equity = _UNIVERSE_MAP.get(ticker)
    if not equity is not None:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    # Generate synthetic price history
    rng = np.random.default_rng(sum(ord(c) for c in ticker))
    dates = list(map(str, _trading_dates(252)))
    initial = equity.close or 100.0
    log_rets = rng.normal(0.0003, 0.015, 252)
    closes = float(initial) * np.exp(np.cumsum(log_rets))

    history = [
        {"date": d, "close": round(float(c), 2), "return": round(float(r), 6)}
        for d, c, r in zip(dates, closes, log_rets)
    ]

    return EquityDetail(**equity.model_dump(), history=history)


def _trading_dates(n: int):
    import pandas as pd
    return pd.bdate_range(end=date.today(), periods=n)
