"""
powerbi/powerbi_export.py
--------------------------
Reads PySpark pipeline outputs (Parquet) and exports
Power BI-ready CSV files with all required columns and relationships.

Run after: 02_feature_engineering notebook (or local equivalent)
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR  = Path("data/raw")
OUTPUT_DIR = Path("powerbi/exports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load(filename: str) -> pd.DataFrame:
    path = INPUT_DIR / filename
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(
        f"'{path}' not found. Run: python data/yahoo_fetcher.py  first."
    )


# ── 1. OHLCV with computed features ─────────────────────────────────────────

def export_ohlcv(ohlcv: pd.DataFrame) -> None:
    """Trim to columns needed by Power BI and add technical indicators."""
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Add SMA columns (Power BI can't do rolling easily)
    for ticker, grp in df.groupby("ticker"):
        idx = grp.index
        df.loc[idx, "sma_20"]  = grp["close"].rolling(20).mean()
        df.loc[idx, "sma_50"]  = grp["close"].rolling(50).mean()
        df.loc[idx, "sma_200"] = grp["close"].rolling(200).mean()

        gains  = grp["daily_return"].clip(lower=0)
        losses = -grp["daily_return"].clip(upper=0)
        avg_g  = gains.rolling(14).mean()
        avg_l  = losses.rolling(14).mean()
        rs     = avg_g / (avg_l + 1e-8)
        df.loc[idx, "rsi_14"] = 100 - (100 / (1 + rs))

        vol = grp["daily_return"].rolling(20).std() * np.sqrt(252)
        df.loc[idx, "vol_20d"] = vol

        cum  = (1 + grp["daily_return"]).cumprod()
        roll = cum.cummax()
        df.loc[idx, "drawdown_pct"] = (cum - roll) / roll * 100

    keep_cols = [
        "ticker","date","open","high","low","close","volume",
        "sector","daily_return","sma_20","sma_50","sma_200",
        "rsi_14","vol_20d","drawdown_pct"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].round(4)

    path = OUTPUT_DIR / "ohlcv_pbi.csv"
    out.to_csv(path, index=False)
    print(f"  OHLCV         → {path}  ({len(out):,} rows)")


# ── 2. Fundamentals ──────────────────────────────────────────────────────────

def export_fundamentals(fund: pd.DataFrame) -> None:
    keep = [
        "ticker","name","sector","sub_industry",
        "beta","market_cap_cat","weight_pct",
        "pe_ratio","dividend_yield","roe","debt_to_equity","market_cap"
    ]
    out = fund[[c for c in keep if c in fund.columns]].copy()
    path = OUTPUT_DIR / "fundamentals_pbi.csv"
    out.to_csv(path, index=False)
    print(f"  Fundamentals  → {path}  ({len(out)} tickers)")


# ── 3. Scenario returns ──────────────────────────────────────────────────────

def export_scenarios(fund: pd.DataFrame) -> None:
    rfr = 0.05
    scenarios = {
        "Bull":         0.25,
        "Base":         0.08,
        "Bear":        -0.30,
        "Stagflation": -0.15,
        "Rate Shock":  -0.10,
    }
    vol_multipliers = {
        "Bull": 0.75, "Base": 1.00, "Bear": 1.80, "Stagflation": 1.40, "Rate Shock": 1.30
    }

    rows = []
    for ticker_row in fund.itertuples():
        beta = float(getattr(ticker_row, "beta", 1.0) or 1.0)
        for scen, mkt in scenarios.items():
            rows.append({
                "ticker":         ticker_row.ticker,
                "sector":         getattr(ticker_row, "sector", ""),
                "scenario_name":  scen,
                "mkt_return_shock": mkt,
                "vol_multiplier":   vol_multipliers[scen],
                "ticker_return":  round(rfr + beta * (mkt - rfr), 4),
                "beta":           round(beta, 3),
            })

    out = pd.DataFrame(rows)
    path = OUTPUT_DIR / "scenario_returns.csv"
    out.to_csv(path, index=False)
    print(f"  Scenarios     → {path}  ({len(out)} rows)")

    # Also write scenario reference table
    scen_ref = pd.DataFrame([
        {"scenario_name": k, "mkt_return_shock": v, "vol_multiplier": vol_multipliers[k]}
        for k, v in scenarios.items()
    ])
    scen_ref.to_csv(OUTPUT_DIR / "scenarios_ref.csv", index=False)


# ── 4. Date table ─────────────────────────────────────────────────────────────

def export_dates(start: str = "2004-01-01") -> None:
    dates = pd.bdate_range(start=start, end=date.today())
    df = pd.DataFrame({
        "date":        dates.date,
        "year":        dates.year,
        "quarter":     dates.quarter,
        "month_num":   dates.month,
        "month_name":  dates.strftime("%b"),
        "week":        dates.isocalendar().week.values,
        "day_of_week": dates.strftime("%a"),
        "is_trading_day": 1,
    })
    path = OUTPUT_DIR / "dates.csv"
    df.to_csv(path, index=False)
    print(f"  Dates         → {path}  ({len(df):,} rows)")


# ── 5. Sector performance summary ────────────────────────────────────────────

def export_sector_summary(ohlcv: pd.DataFrame) -> None:
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.DateOffset(years=1)
    recent = df[df["date"] >= cutoff]

    summary = (
        recent.groupby("sector")
        .agg(
            n_equities    =("ticker", "nunique"),
            avg_return_1y =("daily_return", lambda x: (x.mean() * 252)),
            volatility    =("daily_return", lambda x: (x.std() * np.sqrt(252))),
            win_rate      =("daily_return", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    summary["sharpe"] = (summary["avg_return_1y"] - 0.05) / summary["volatility"]
    summary = summary.round(4)

    path = OUTPUT_DIR / "sector_summary.csv"
    summary.to_csv(path, index=False)
    print(f"  Sector summary→ {path}  ({len(summary)} sectors)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\nPower BI Export — {date.today()}")
    print("="*50)

    try:
        ohlcv = _load("ohlcv.parquet")
        fund  = _load("fundamentals.parquet")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nGenerate sample data first:")
        print("  python data/yahoo_fetcher.py --n-tickers 50 --years 5")
        return

    print(f"Loaded: {len(ohlcv):,} OHLCV rows, {ohlcv['ticker'].nunique()} tickers\n")

    export_ohlcv(ohlcv)
    export_fundamentals(fund)
    export_scenarios(fund)
    export_dates()
    export_sector_summary(ohlcv)

    print(f"\n✓ All exports written to {OUTPUT_DIR.resolve()}/")
    print("\nNext: Open Power BI Desktop → Get Data → Text/CSV → import all files")
    print("      Then follow: powerbi/README.md for setup instructions")


if __name__ == "__main__":
    run()
