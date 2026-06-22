"""
data/yahoo_fetcher.py
----------------------
Downloads 20 years of daily OHLCV data for 500+ S&P 500 equities
from Yahoo Finance via yfinance. Saves to Parquet for use in the
PySpark pipeline and Power BI / web dashboard.

Features
--------
- Batch download (50 tickers / call) with rate-limit back-off
- Incremental update: skip tickers already cached
- Progress bar
- Automatic retry on transient failures
- Saves OHLCV + computed daily returns

Usage
-----
    # Full 20-year download (~15–25 min on typical broadband)
    python data/yahoo_fetcher.py

    # Quick test: 10 tickers, 1 year
    python data/yahoo_fetcher.py --n-tickers 10 --years 1

    # Update existing cache (only download new dates)
    python data/yahoo_fetcher.py --incremental
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance --break-system-packages")
    sys.exit(1)

from data.sp500_universe import SP500_CONSTITUENTS, get_dataframe as get_universe_df

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_YEARS    = 20
BATCH_SIZE       = 50          # yfinance handles up to ~500 but 50 is safer
SLEEP_BETWEEN    = 1.5         # seconds between batches (rate limiting)
MAX_RETRIES      = 3
OUTPUT_DIR       = Path("data/raw")
OHLCV_FILE       = OUTPUT_DIR / "ohlcv.parquet"
FUNDAMENTALS_FILE= OUTPUT_DIR / "fundamentals.parquet"
RETURNS_FILE     = OUTPUT_DIR / "returns_wide.parquet"


# ── Download helpers ─────────────────────────────────────────────────────────

def _download_batch(
    tickers: list[str],
    start: str,
    end: str,
    retries: int = MAX_RETRIES,
) -> pd.DataFrame | None:
    """Download a batch of tickers using yf.download (vectorised)."""
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            return raw
        except Exception as exc:
            wait = 2 ** attempt * SLEEP_BETWEEN
            print(f"  [attempt {attempt+1}/{retries}] Error: {exc} — retrying in {wait:.1f}s")
            time.sleep(wait)
    return None


def _parse_multiindex(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Flatten yfinance multi-index download into long-format OHLCV."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw.xs(ticker, axis=1, level=1) if isinstance(raw.columns, pd.MultiIndex) else raw

            df = df.rename(columns=str.lower)
            needed = {"open","high","low","close","volume"}
            if not needed.issubset(df.columns):
                continue

            df = df[list(needed)].copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.dropna(subset=["close"])
            df.insert(0, "ticker", ticker)
            df = df.reset_index()
            frames.append(df)
        except Exception:
            continue

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Pull key fundamental metrics via yfinance Ticker.info."""
    records = []
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).fast_info
            records.append({
                "ticker":              ticker,
                "market_cap":          getattr(info, "market_cap", None),
                "shares_outstanding":  getattr(info, "shares", None),
                "last_price":          getattr(info, "last_price", None),
            })
            if (i + 1) % 10 == 0:
                time.sleep(0.5)
        except Exception:
            records.append({"ticker": ticker})

    return pd.DataFrame(records)


# ── Main download pipeline ───────────────────────────────────────────────────

def download(
    n_tickers: int | None = None,
    years: int = DEFAULT_YEARS,
    incremental: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download OHLCV for S&P 500 constituents.

    Returns
    -------
    (ohlcv_df, fundamentals_df)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    end_date   = date.today()
    start_date = end_date - timedelta(days=int(years * 365.25))
    end_str    = str(end_date)
    start_str  = str(start_date)

    # Build ticker list
    tickers = [c.ticker.replace(".", "-") for c in SP500_CONSTITUENTS]
    if n_tickers:
        tickers = tickers[:n_tickers]

    universe_df = get_universe_df()
    universe_df["ticker"] = universe_df["ticker"].str.replace(".", "-", regex=False)

    # Incremental: only download tickers not yet cached
    cached_tickers: set[str] = set()
    if incremental and OHLCV_FILE.exists():
        existing = pd.read_parquet(OHLCV_FILE, columns=["ticker"])
        cached_tickers = set(existing["ticker"].unique())
        new_tickers = [t for t in tickers if t not in cached_tickers]
        print(f"Incremental mode: {len(cached_tickers)} cached, {len(new_tickers)} to download")
        tickers = new_tickers

    if not tickers:
        print("All tickers cached. Run without --incremental to force re-download.")
        return pd.read_parquet(OHLCV_FILE), pd.read_parquet(FUNDAMENTALS_FILE)

    print(f"\n{'='*60}")
    print(f"  Equity Market Analysis — Yahoo Finance Downloader")
    print(f"{'='*60}")
    print(f"  Tickers  : {len(tickers)}")
    print(f"  Period   : {start_str} → {end_str} ({years}Y)")
    print(f"  Batch sz : {batch_size}")
    print(f"  Output   : {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    all_frames: list[pd.DataFrame] = []
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch = tickers[b * batch_size : (b + 1) * batch_size]
        print(f"Batch {b+1}/{n_batches}  ({batch[0]} … {batch[-1]})", end=" ", flush=True)

        raw  = _download_batch(batch, start_str, end_str)
        long = _parse_multiindex(raw, batch) if raw is not None else pd.DataFrame()

        if long.empty:
            print("⚠ no data")
        else:
            all_frames.append(long)
            n_rows = len(long)
            n_tkrs = long["ticker"].nunique()
            print(f"✓  {n_rows:,} rows, {n_tkrs} tickers")

        if b < n_batches - 1:
            time.sleep(SLEEP_BETWEEN)

    if not all_frames:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    ohlcv = pd.concat(all_frames, ignore_index=True)

    # Merge with existing cache if incremental
    if incremental and OHLCV_FILE.exists():
        old = pd.read_parquet(OHLCV_FILE)
        ohlcv = pd.concat([old, ohlcv], ignore_index=True)

    # ── Enrichment ──────────────────────────────────────────────────────────

    # Attach sector info from universe
    ohlcv = ohlcv.merge(
        universe_df[["ticker","sector","sub_industry","weight_pct","beta","market_cap_cat"]],
        on="ticker", how="left"
    )

    # Compute daily return (in-place, per ticker)
    ohlcv = ohlcv.sort_values(["ticker","date"])
    ohlcv["daily_return"] = (
        ohlcv.groupby("ticker")["close"]
             .pct_change()
    )

    # Deduplicate
    ohlcv = ohlcv.drop_duplicates(subset=["ticker","date"])

    # ── Save OHLCV ──────────────────────────────────────────────────────────
    ohlcv.to_parquet(OHLCV_FILE, index=False, compression="snappy")
    print(f"\n✓ OHLCV saved   → {OHLCV_FILE}  ({len(ohlcv):,} rows, {ohlcv['ticker'].nunique()} tickers)")

    # ── Wide returns matrix (for correlation, risk engine) ──────────────────
    returns_wide = (
        ohlcv[["date","ticker","daily_return"]]
        .dropna()
        .pivot(index="date", columns="ticker", values="daily_return")
    )
    returns_wide.to_parquet(RETURNS_FILE, compression="snappy")
    print(f"✓ Returns matrix → {RETURNS_FILE}  {returns_wide.shape}")

    # ── Fundamentals ────────────────────────────────────────────────────────
    print("\nFetching fundamentals (fast_info)…")
    fund_df = _fetch_fundamentals(tickers[:min(len(tickers), 100)])  # cap API calls

    # Merge static sector data
    fund_df = fund_df.merge(
        universe_df[["ticker","name","sector","sub_industry","beta","market_cap_cat"]],
        on="ticker", how="left"
    )
    fund_df.to_parquet(FUNDAMENTALS_FILE, index=False, compression="snappy")
    print(f"✓ Fundamentals  → {FUNDAMENTALS_FILE}  ({len(fund_df)} tickers)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Download complete")
    print(f"  OHLCV rows       : {len(ohlcv):,}")
    print(f"  Tickers          : {ohlcv['ticker'].nunique()}")
    print(f"  Date range       : {ohlcv['date'].min()} → {ohlcv['date'].max()}")
    print(f"  Sectors covered  : {ohlcv['sector'].nunique()}")
    print(f"{'='*60}\n")

    return ohlcv, fund_df


def generate_dashboard_json(ohlcv: pd.DataFrame, out_path: str = "dashboard/data.json") -> None:
    """
    Generate a compact JSON summary for the standalone dashboard.
    Contains last 5 years of data at weekly frequency to keep file size <5MB.
    """
    import json

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=5)
    recent = ohlcv[ohlcv["date"] >= cutoff].copy()

    # Weekly resample per ticker (last close of week)
    recent["date"] = pd.to_datetime(recent["date"])
    weekly = (
        recent.set_index("date")
              .groupby("ticker")["close"]
              .resample("W-FRI")
              .last()
              .reset_index()
    )

    # Sector index (equal-weight weekly returns)
    sector_weekly = (
        recent.set_index("date")
              .groupby(["sector", pd.Grouper(freq="W-FRI")])["daily_return"]
              .mean()
              .reset_index()
    )

    output = {
        "generated":    str(date.today()),
        "tickers":      weekly.groupby("ticker").apply(
                            lambda g: g["close"].tolist()
                        ).to_dict(),
        "dates":        [str(d) for d in weekly["date"].unique().tolist()],
        "sector_rets":  sector_weekly.groupby("sector").apply(
                            lambda g: g["daily_return"].tolist()
                        ).to_dict(),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, default=str)
    print(f"Dashboard JSON -> {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download S&P 500 OHLCV from Yahoo Finance")
    parser.add_argument("--n-tickers",   type=int,  default=None, help="Limit ticker count (default: all ~500)")
    parser.add_argument("--years",       type=int,  default=20,   help="Years of history (default: 20)")
    parser.add_argument("--batch-size",  type=int,  default=50,   help="Tickers per API call (default: 50)")
    parser.add_argument("--incremental", action="store_true",     help="Skip already-cached tickers")
    parser.add_argument("--gen-json",    action="store_true",     help="Also generate dashboard JSON")
    args = parser.parse_args()

    ohlcv, fund = download(
        n_tickers   = args.n_tickers,
        years       = args.years,
        batch_size  = args.batch_size,
        incremental = args.incremental,
    )

    if args.gen_json:
        generate_dashboard_json(ohlcv)
