"""
sample_generator.py
-------------------
Generates a realistic 500-equity dataset with 5 years of daily OHLCV data,
fundamental metrics, and sector classifications. Output saved as Parquet
(columnar, fast) and CSV (human-readable).

Usage:
    python data/sample_generator.py [--n-equities 500] [--years 5]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTORS = {
    "Technology": 0.25,
    "Financials": 0.18,
    "Healthcare": 0.14,
    "Consumer Discretionary": 0.11,
    "Industrials": 0.10,
    "Communication Services": 0.08,
    "Consumer Staples": 0.06,
    "Energy": 0.04,
    "Materials": 0.02,
    "Utilities": 0.01,
    "Real Estate": 0.01,
}

MARKET_CAPS = ["Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
EXCHANGES = ["NYSE", "NASDAQ", "LSE", "NSE"]

SECTOR_BETAS = {
    "Technology": (1.15, 0.20),
    "Financials": (1.05, 0.15),
    "Healthcare": (0.85, 0.18),
    "Consumer Discretionary": (1.10, 0.20),
    "Industrials": (1.00, 0.15),
    "Communication Services": (0.95, 0.18),
    "Consumer Staples": (0.65, 0.12),
    "Energy": (1.20, 0.25),
    "Materials": (1.10, 0.20),
    "Utilities": (0.55, 0.10),
    "Real Estate": (0.80, 0.15),
}


# ---------------------------------------------------------------------------
# Ticker generation
# ---------------------------------------------------------------------------

def _make_tickers(n: int) -> list[str]:
    """Generate unique synthetic tickers (e.g. ABCD)."""
    rng = np.random.default_rng(42)
    alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    seen: set[str] = set()
    tickers: list[str] = []
    while len(tickers) < n:
        length = rng.integers(3, 5)
        t = "".join(rng.choice(alpha, length))
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers


# ---------------------------------------------------------------------------
# OHLCV data
# ---------------------------------------------------------------------------

def _simulate_ohlcv(
    ticker: str,
    sector: str,
    beta: float,
    start: str,
    end: str,
    seed: int,
) -> pd.DataFrame:
    """Simulate daily OHLCV using GBM with sector-correlated drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Drift and volatility
    annual_return = rng.normal(0.08 * beta, 0.04)
    annual_vol = rng.uniform(0.15, 0.45)
    dt = 1 / 252

    # GBM log returns
    drift = (annual_return - 0.5 * annual_vol**2) * dt
    diffusion = annual_vol * np.sqrt(dt)
    log_returns = rng.normal(drift, diffusion, n)

    # Market shock events (3 random shock days per year on average)
    shock_days = rng.choice(n, size=int(n * 0.012), replace=False)
    log_returns[shock_days] += rng.normal(0, 0.03, len(shock_days))

    # Price series
    initial_price = rng.uniform(10, 2000)
    closes = initial_price * np.exp(np.cumsum(log_returns))

    # OHLV from close
    daily_range = rng.uniform(0.005, 0.025, n)
    highs = closes * (1 + daily_range)
    lows = closes * (1 - daily_range)
    opens = lows + rng.random(n) * (highs - lows)
    volumes = rng.integers(100_000, 50_000_000, n).astype(float)
    volumes *= 1 + 0.5 * np.abs(log_returns)  # higher vol on big moves

    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volumes.astype(int),
        "sector": sector,
    })


# ---------------------------------------------------------------------------
# Fundamental metrics
# ---------------------------------------------------------------------------

def _generate_fundamentals(tickers: list[str], sectors: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = len(tickers)

    pe_by_sector = {
        "Technology": (28, 10),
        "Financials": (14, 5),
        "Healthcare": (22, 8),
        "Consumer Discretionary": (20, 7),
        "Industrials": (18, 6),
        "Communication Services": (25, 9),
        "Consumer Staples": (22, 5),
        "Energy": (12, 4),
        "Materials": (15, 5),
        "Utilities": (17, 4),
        "Real Estate": (30, 10),
    }

    pe_ratios, pb_ratios, div_yields, roe, debt_equity = [], [], [], [], []
    for sec in sectors:
        mu, sigma = pe_by_sector.get(sec, (20, 7))
        pe_ratios.append(max(rng.normal(mu, sigma), 3.0))
        pb_ratios.append(max(rng.normal(2.5, 1.0), 0.5))
        div_yields.append(max(rng.normal(0.02, 0.015), 0.0))
        roe.append(rng.normal(0.14, 0.07))
        debt_equity.append(max(rng.normal(0.6, 0.4), 0.0))

    market_caps = rng.choice(MARKET_CAPS, n, p=[0.40, 0.30, 0.20, 0.10])
    exchanges = rng.choice(EXCHANGES, n, p=[0.40, 0.35, 0.15, 0.10])
    country = ["US" if ex in ("NYSE", "NASDAQ") else ("UK" if ex == "LSE" else "IN")
               for ex in exchanges]

    betas = [rng.normal(*SECTOR_BETAS.get(sec, (1.0, 0.15))) for sec in sectors]

    return pd.DataFrame({
        "ticker": tickers,
        "sector": sectors,
        "market_cap_category": market_caps,
        "exchange": exchanges,
        "country": country,
        "beta": np.round(betas, 3),
        "pe_ratio": np.round(pe_ratios, 2),
        "pb_ratio": np.round(pb_ratios, 2),
        "dividend_yield": np.round(div_yields, 4),
        "roe": np.round(roe, 4),
        "debt_to_equity": np.round(debt_equity, 3),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n_equities: int = 500, years: int = 5, out_dir: str = "data/raw") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)

    tickers = _make_tickers(n_equities)

    # Assign sectors proportionally
    sector_names = list(SECTORS.keys())
    sector_probs = list(SECTORS.values())
    rng = np.random.default_rng(0)
    sectors = rng.choice(sector_names, size=n_equities, p=sector_probs).tolist()

    print(f"Generating OHLCV data for {n_equities} equities ({start.date()} to {end.date()})...")
    ohlcv_frames = []
    for i, (ticker, sector) in enumerate(zip(tickers, sectors)):
        beta_mu, _ = SECTOR_BETAS.get(sector, (1.0, 0.15))
        beta = float(np.random.default_rng(i).normal(beta_mu, 0.15))
        ohlcv_frames.append(_simulate_ohlcv(ticker, sector, beta, str(start.date()), str(end.date()), seed=i))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_equities} done")

    ohlcv = pd.concat(ohlcv_frames, ignore_index=True)
    fundamentals = _generate_fundamentals(tickers, sectors)

    # Save
    ohlcv_path = Path(out_dir) / "ohlcv.parquet"
    fund_path = Path(out_dir) / "fundamentals.parquet"
    ohlcv_csv = Path(out_dir) / "ohlcv_sample.csv"
    fund_csv = Path(out_dir) / "fundamentals.csv"

    ohlcv.to_parquet(ohlcv_path, index=False, compression="snappy")
    fundamentals.to_parquet(fund_path, index=False, compression="snappy")

    # CSV sample (last 252 trading days only, to keep it small)
    recent = ohlcv[ohlcv["date"] >= end - pd.DateOffset(years=1)]
    recent.to_csv(ohlcv_csv, index=False)
    fundamentals.to_csv(fund_csv, index=False)

    print(f"\nDataset summary:")
    print(f"  OHLCV rows   : {len(ohlcv):,}  -> {ohlcv_path}")
    print(f"  Fundamentals : {len(fundamentals):,}  -> {fund_path}")
    print(f"  CSV samples  : {ohlcv_csv}, {fund_csv}")
    print(f"\nSector distribution:")
    for sec, cnt in fundamentals["sector"].value_counts().items():
        print(f"  {sec:<30} {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate equity sample dataset")
    parser.add_argument("--n-equities", type=int, default=500)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="data/raw")
    args = parser.parse_args()
    generate(args.n_equities, args.years, args.out_dir)
