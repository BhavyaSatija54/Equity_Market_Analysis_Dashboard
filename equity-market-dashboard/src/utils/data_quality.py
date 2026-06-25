"""
src/utils/data_quality.py
--------------------------
Named, configurable Data Quality Engine used across all
medallion layers (Bronze → Silver → Gold).

Every layer calls DataQualityEngine.run_checks() before writing.
Results are logged as structured JSON (ELK/Datadog compatible)
and surfaced in the pipeline run report.

Configurable via config/pipeline_config.yaml:
  quality:
    null_threshold_pct:    0.01   # 1% max nulls per column
    price_move_limit_pct:  0.50   # 50% daily move = circuit breaker
    freshness_hours:       48     # data must be within 48h
    strict_mode:           true   # fail pipeline on breach
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check_name: str
    layer: str                   # bronze | silver | gold
    status: str                  # PASS | WARN | FAIL
    metric: Optional[float] = None
    threshold: Optional[float] = None
    detail: str = ""
    affected_rows: int = 0
    affected_cols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "check_name":    self.check_name,
            "layer":         self.layer,
            "status":        self.status,
            "metric":        round(self.metric, 6) if self.metric is not None else None,
            "threshold":     self.threshold,
            "detail":        self.detail,
            "affected_rows": self.affected_rows,
            "affected_cols": self.affected_cols,
        }


@dataclass
class QualityReport:
    layer: str
    run_id: str
    timestamp: str
    total_rows: int
    total_cols: int
    checks: list[CheckResult] = field(default_factory=list)
    overall_score: float = 100.0   # 0–100

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "PASS")

    @property
    def warned(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    def to_dict(self) -> dict:
        return {
            "layer":          self.layer,
            "run_id":         self.run_id,
            "timestamp":      self.timestamp,
            "total_rows":     self.total_rows,
            "total_cols":     self.total_cols,
            "overall_score":  round(self.overall_score, 2),
            "summary":        {"pass": self.passed, "warn": self.warned, "fail": self.failed},
            "checks":         [c.to_dict() for c in self.checks],
        }

    def log_structured(self) -> None:
        """Emit structured JSON log for ELK / Datadog ingestion."""
        logger.info(json.dumps(self.to_dict()))


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class QualityConfig:
    null_threshold_pct: float = 0.01      # Max fraction of nulls per column
    price_move_limit_pct: float = 0.50    # Circuit breaker: flag moves > 50%
    freshness_hours: int = 48             # Data must be within N hours
    min_rows: int = 1                     # Minimum acceptable row count
    strict_mode: bool = True              # True → raise on FAIL; False → warn only
    required_columns: list[str] = field(default_factory=lambda: [
        "ticker", "date", "close"
    ])
    numeric_range_checks: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "close":  (0.01, 1_000_000),
        "open":   (0.01, 1_000_000),
        "high":   (0.01, 1_000_000),
        "low":    (0.01, 1_000_000),
        "volume": (0,    1e12),
    })

    @classmethod
    def from_dict(cls, cfg: dict) -> "QualityConfig":
        return cls(
            null_threshold_pct    = cfg.get("null_threshold_pct",    0.01),
            price_move_limit_pct  = cfg.get("price_move_limit_pct",  0.50),
            freshness_hours       = cfg.get("freshness_hours",        48),
            min_rows              = cfg.get("min_rows",               1),
            strict_mode           = cfg.get("strict_mode",            True),
            required_columns      = cfg.get("required_columns",       ["ticker","date","close"]),
        )


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────

class DataQualityEngine:
    """
    Configurable data quality engine for the medallion pipeline.

    Usage
    -----
    engine = DataQualityEngine(layer="bronze", config=QualityConfig())
    report = engine.run_checks(df)
    # raises DataQualityError if strict_mode=True and any check fails
    """

    def __init__(
        self,
        layer: str,
        config: Optional[QualityConfig] = None,
        run_id: Optional[str] = None,
    ):
        self.layer  = layer
        self.cfg    = config or QualityConfig()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Public API ───────────────────────────────────────────

    def run_checks(self, df: pd.DataFrame) -> QualityReport:
        """
        Run all configured checks against df.
        Returns a QualityReport; raises DataQualityError if
        strict_mode=True and any check has status=FAIL.
        """
        report = QualityReport(
            layer     = self.layer,
            run_id    = self.run_id,
            timestamp = datetime.now().isoformat(),
            total_rows= len(df),
            total_cols= len(df.columns),
        )

        # Run every check method
        check_methods = [
            self._check_row_count,
            self._check_required_columns,
            self._check_null_rates,
            self._check_numeric_ranges,
            self._check_ohlc_consistency,
            self._check_price_moves,
            self._check_duplicates,
            self._check_freshness,
        ]

        for method in check_methods:
            try:
                result = method(df)
                if result:
                    report.checks.append(result)
            except Exception as exc:
                logger.warning(f"[DQ] Check {method.__name__} raised: {exc}")
                report.checks.append(CheckResult(
                    check_name=method.__name__, layer=self.layer,
                    status="WARN", detail=str(exc)
                ))

        # Score: deduct 10 per FAIL, 3 per WARN
        report.overall_score = max(
            0.0,
            100.0
            - report.failed * 10
            - report.warned * 3
        )

        report.log_structured()
        self._summary_log(report)

        if self.cfg.strict_mode and report.failed > 0:
            failed_names = [c.check_name for c in report.checks if c.status == "FAIL"]
            raise DataQualityError(
                f"[{self.layer.upper()}] {report.failed} DQ check(s) failed: {failed_names}"
            )

        return report

    # ── Individual checks ────────────────────────────────────

    def _check_row_count(self, df: pd.DataFrame) -> CheckResult:
        status = "PASS" if len(df) >= self.cfg.min_rows else "FAIL"
        return CheckResult(
            check_name   = "row_count",
            layer        = self.layer,
            status       = status,
            metric       = float(len(df)),
            threshold    = float(self.cfg.min_rows),
            detail       = f"{len(df):,} rows (min={self.cfg.min_rows:,})",
            affected_rows= 0 if status == "PASS" else len(df),
        )

    def _check_required_columns(self, df: pd.DataFrame) -> CheckResult:
        missing = [c for c in self.cfg.required_columns if c not in df.columns]
        return CheckResult(
            check_name   = "required_columns",
            layer        = self.layer,
            status       = "FAIL" if missing else "PASS",
            detail       = f"Missing columns: {missing}" if missing else "All required columns present",
            affected_cols= missing,
        )

    def _check_null_rates(self, df: pd.DataFrame) -> CheckResult:
        null_rates = (df.isnull().sum() / max(len(df), 1)).sort_values(ascending=False)
        breaches   = null_rates[null_rates > self.cfg.null_threshold_pct]
        status     = "FAIL" if len(breaches) > 0 else "PASS"
        worst      = float(null_rates.iloc[0]) if len(null_rates) > 0 else 0.0
        return CheckResult(
            check_name   = "null_rates",
            layer        = self.layer,
            status       = status,
            metric       = worst,
            threshold    = self.cfg.null_threshold_pct,
            detail       = f"{len(breaches)} column(s) exceed null threshold: {list(breaches.index)}",
            affected_cols= list(breaches.index),
        )

    def _check_numeric_ranges(self, df: pd.DataFrame) -> CheckResult:
        violations: list[str] = []
        for col, (lo, hi) in self.cfg.numeric_range_checks.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            out_of_range = ((series < lo) | (series > hi)).sum()
            if out_of_range > 0:
                violations.append(f"{col}:{out_of_range}")
        return CheckResult(
            check_name   = "numeric_ranges",
            layer        = self.layer,
            status       = "FAIL" if violations else "PASS",
            detail       = f"Range violations: {violations}" if violations else "All numeric ranges valid",
            affected_cols= [v.split(":")[0] for v in violations],
        )

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> CheckResult:
        needed = {"open", "high", "low", "close"}
        if not needed.issubset(df.columns):
            return CheckResult(check_name="ohlc_consistency", layer=self.layer, status="PASS",
                               detail="OHLC columns not present; skipped")
        mask = (
            (df["high"] < df["low"])
            | (df["close"] > df["high"] * 1.001)
            | (df["close"] < df["low"]  * 0.999)
            | (df["open"]  > df["high"] * 1.001)
            | (df["open"]  < df["low"]  * 0.999)
        )
        n = int(mask.sum())
        return CheckResult(
            check_name   = "ohlc_consistency",
            layer        = self.layer,
            status       = "FAIL" if n > 0 else "PASS",
            metric       = float(n),
            detail       = f"{n} rows violate OHLC ordering (high≥close≥low)",
            affected_rows= n,
        )

    def _check_price_moves(self, df: pd.DataFrame) -> CheckResult:
        """Circuit breaker: flag extreme single-day moves."""
        if "daily_return" not in df.columns:
            if "close" not in df.columns:
                return CheckResult(check_name="price_moves", layer=self.layer,
                                   status="PASS", detail="No return column; skipped")
            df = df.copy()
            df["daily_return"] = df.groupby("ticker")["close"].pct_change() if "ticker" in df.columns \
                                  else df["close"].pct_change()

        extreme = (df["daily_return"].abs() > self.cfg.price_move_limit_pct).sum()
        pct_extreme = extreme / max(len(df), 1)
        status = "WARN" if pct_extreme > 0.001 else "PASS"   # >0.1% rows = warn
        return CheckResult(
            check_name   = "price_moves",
            layer        = self.layer,
            status       = status,
            metric       = float(pct_extreme),
            threshold    = self.cfg.price_move_limit_pct,
            detail       = f"{int(extreme)} rows exceed {self.cfg.price_move_limit_pct:.0%} daily move limit",
            affected_rows= int(extreme),
        )

    def _check_duplicates(self, df: pd.DataFrame) -> CheckResult:
        key_cols = [c for c in ["ticker", "date"] if c in df.columns]
        if not key_cols:
            return CheckResult(check_name="duplicates", layer=self.layer,
                               status="PASS", detail="No key columns; skipped")
        dupes = int(df.duplicated(subset=key_cols).sum())
        return CheckResult(
            check_name   = "duplicates",
            layer        = self.layer,
            status       = "FAIL" if dupes > 0 else "PASS",
            metric       = float(dupes),
            detail       = f"{dupes} duplicate (ticker, date) pairs",
            affected_rows= dupes,
        )

    def _check_freshness(self, df: pd.DataFrame) -> CheckResult:
        if "date" not in df.columns:
            return CheckResult(check_name="freshness", layer=self.layer,
                               status="PASS", detail="No date column; skipped")
        try:
            max_date = pd.to_datetime(df["date"]).max()
            cutoff   = datetime.now() - timedelta(hours=self.cfg.freshness_hours)
            is_fresh = max_date >= pd.Timestamp(cutoff)
            hours_old = (datetime.now() - max_date.to_pydatetime()).total_seconds() / 3600
            return CheckResult(
                check_name = "freshness",
                layer      = self.layer,
                status     = "PASS" if is_fresh else "WARN",   # WARN not FAIL for stale data
                metric     = round(hours_old, 1),
                threshold  = float(self.cfg.freshness_hours),
                detail     = f"Latest date: {max_date.date()} ({hours_old:.1f}h ago)",
            )
        except Exception as exc:
            return CheckResult(check_name="freshness", layer=self.layer,
                               status="WARN", detail=str(exc))

    # ── Helpers ──────────────────────────────────────────────

    def _summary_log(self, report: QualityReport) -> None:
        score_icon = "✓" if report.overall_score >= 90 else "⚠" if report.overall_score >= 70 else "✗"
        logger.info(
            f"[DQ] [{self.layer.upper()}] {score_icon} Score={report.overall_score:.1f} "
            f"| PASS={report.passed} WARN={report.warned} FAIL={report.failed} "
            f"| rows={report.total_rows:,}"
        )


# ─────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────

class DataQualityError(Exception):
    """Raised when strict_mode=True and a DQ check fails."""
    pass
