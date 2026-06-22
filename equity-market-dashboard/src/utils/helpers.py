"""
src/utils/helpers.py
---------------------
Shared utility functions: config loading, date helpers, number formatting.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """Load YAML config with environment variable substitution."""
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Config not found at {path}; using defaults")
        return {}
    with open(config_path) as f:
        raw = f.read()

    # Simple env-var substitution: ${VAR:default}
    import re
    def replace(m):
        var, *rest = m.group(1).split(":")
        default = rest[0] if rest else ""
        return os.environ.get(var, default)

    substituted = re.sub(r"\$\{([^}]+)\}", replace, raw)
    return yaml.safe_load(substituted)


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def trading_dates(start: date, end: date) -> list[date]:
    """Return list of weekday (Mon–Fri) dates between start and end inclusive."""
    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def n_trading_days_ago(n: int, from_date: date | None = None) -> date:
    """Return the date that was ~n trading days before from_date."""
    ref = from_date or date.today()
    total_days = int(n * (7 / 5)) + 10   # generous buffer for weekends
    candidate  = ref - timedelta(days=total_days)
    tds = trading_dates(candidate, ref)
    return tds[max(0, len(tds) - n)]


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def fmt_pct(v: float, decimals: int = 2, signed: bool = True) -> str:
    sign = "+" if signed and v >= 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"


def fmt_large(v: float) -> str:
    """Format large numbers with K/M/B suffix."""
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.2f}K"
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# Sector utilities
# ---------------------------------------------------------------------------

SECTOR_COLORS: dict[str, str] = {
    "Technology":             "#58a6ff",
    "Financials":             "#ffc400",
    "Healthcare":             "#3fb950",
    "Consumer Discretionary": "#f85149",
    "Industrials":            "#a15aff",
    "Communication Services": "#39c5cf",
    "Consumer Staples":       "#fb8500",
    "Energy":                 "#e63946",
    "Materials":              "#8ecae6",
    "Utilities":              "#b5838d",
    "Real Estate":            "#ccd5ae",
}


def sector_color(sector: str) -> str:
    return SECTOR_COLORS.get(sector, "#8b949e")


# ---------------------------------------------------------------------------
# Retry decorator (for external data calls)
# ---------------------------------------------------------------------------

def with_retry(max_attempts: int = 3, wait: float = 1.0):
    from tenacity import retry, stop_after_attempt, wait_fixed
    return retry(stop=stop_after_attempt(max_attempts), wait=wait_fixed(wait))
