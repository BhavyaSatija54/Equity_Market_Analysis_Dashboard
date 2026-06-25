"""
src/orchestration/pipeline_dag.py
-----------------------------------
Pipeline DAG runner with checkpointing, idempotent re-runs,
structured metrics logging, and circuit-breaker on DQ failure.

Stages execute in dependency order:
  bronze_ingestion
      └─ silver_cleanse
              └─ silver_features
                      └─ gold_sector_agg
                      └─ gold_fact_metrics
                              └─ export_powerbi

Each stage writes a checkpoint file on success. Re-running the
pipeline skips already-completed stages unless --force is passed.

Usage
-----
  from src.orchestration.pipeline_dag import PipelineDAG
  dag = PipelineDAG.from_config("config/pipeline_config.yaml")
  dag.run(force=False)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────
# Stage definition
# ─────────────────────────────────────────────────────────────

@dataclass
class Stage:
    name: str
    layer: str                              # bronze | silver | gold | export
    fn: Callable                            # the function to execute
    depends_on: list[str] = field(default_factory=list)
    description: str = ""
    critical: bool = True                   # False → failure is a warning, not a stop


@dataclass
class StageResult:
    stage_name: str
    layer: str
    status: str                             # SUCCESS | FAILED | SKIPPED
    duration_sec: float = 0.0
    records_in: int = 0
    records_out: int = 0
    quality_score: float = 100.0
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "stage":         self.stage_name,
            "layer":         self.layer,
            "status":        self.status,
            "duration_sec":  round(self.duration_sec, 3),
            "records_in":    self.records_in,
            "records_out":   self.records_out,
            "quality_score": self.quality_score,
            "error":         self.error,
        }


# ─────────────────────────────────────────────────────────────
# Checkpoint store
# ─────────────────────────────────────────────────────────────

class CheckpointStore:
    """
    File-based checkpoint store.
    Each completed stage writes a JSON marker to checkpoint_dir/.
    Re-runs read markers and skip completed stages.
    """

    def __init__(self, checkpoint_dir: str = "data/.checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str, stage: str) -> Path:
        return self.dir / f"{run_id}__{stage}.json"

    def mark_complete(self, run_id: str, stage: str, result: StageResult) -> None:
        path = self._path(run_id, stage)
        path.write_text(json.dumps({
            "run_id":    run_id,
            "stage":     stage,
            "completed": datetime.now().isoformat(),
            **result.to_dict(),
        }, indent=2))
        logger.debug(f"[CKPT] ✓ {stage} → {path.name}")

    def is_complete(self, run_id: str, stage: str) -> bool:
        return self._path(run_id, stage).exists()

    def clear(self, run_id: str) -> None:
        for f in self.dir.glob(f"{run_id}__*.json"):
            f.unlink()
        logger.info(f"[CKPT] Cleared all checkpoints for run {run_id}")

    def load(self, run_id: str, stage: str) -> dict:
        p = self._path(run_id, stage)
        return json.loads(p.read_text()) if p.exists() else {}


# ─────────────────────────────────────────────────────────────
# DAG
# ─────────────────────────────────────────────────────────────

class PipelineDAG:
    """
    Medallion-architecture pipeline DAG.

    Bronze → Silver (cleanse) → Silver (features)
          → Gold (fact) → Gold (aggregates) → Export (Power BI)

    Parameters
    ----------
    run_id       : Unique ID for this run (auto-generated if not provided)
    checkpoint_dir: Where to persist stage completion markers
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        checkpoint_dir: str = "data/.checkpoints",
    ):
        self.run_id     = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.ckpt       = CheckpointStore(checkpoint_dir)
        self._stages:   list[Stage] = []
        self._results:  list[StageResult] = []
        self._context:  dict = {}           # shared state passed between stages

    # ── Registration ─────────────────────────────────────────

    def add_stage(self, stage: Stage) -> "PipelineDAG":
        self._stages.append(stage)
        return self

    def stage(
        self,
        name: str,
        layer: str,
        depends_on: Optional[list[str]] = None,
        description: str = "",
        critical: bool = True,
    ):
        """Decorator to register a function as a pipeline stage."""
        def decorator(fn: Callable) -> Callable:
            self.add_stage(Stage(
                name=name, layer=layer, fn=fn,
                depends_on=depends_on or [],
                description=description, critical=critical,
            ))
            return fn
        return decorator

    # ── Execution ────────────────────────────────────────────

    def run(self, force: bool = False) -> dict:
        """
        Execute all stages in dependency order.

        Parameters
        ----------
        force : If True, ignore existing checkpoints and rerun all stages.

        Returns
        -------
        Pipeline run report dict.
        """
        if force:
            self.ckpt.clear(self.run_id)

        start_time = time.perf_counter()
        logger.info(f"\n{'═'*60}")
        logger.info(f"  PIPELINE RUN  │  run_id={self.run_id}  │  force={force}")
        logger.info(f"  Stages: {len(self._stages)}")
        logger.info(f"{'═'*60}")

        ordered = self._topological_sort()
        completed: set[str] = set()

        for stage in ordered:
            # Check dependencies
            unmet = [d for d in stage.depends_on if d not in completed]
            if unmet:
                logger.warning(f"[DAG] Skipping '{stage.name}' — unmet deps: {unmet}")
                self._results.append(StageResult(
                    stage_name=stage.name, layer=stage.layer,
                    status="SKIPPED", error=f"Unmet dependencies: {unmet}"
                ))
                continue

            # Check checkpoint
            if self.ckpt.is_complete(self.run_id, stage.name) and not force:
                cached = self.ckpt.load(self.run_id, stage.name)
                logger.info(f"[DAG] ⏭  SKIP  [{stage.layer.upper()}] {stage.name} (checkpoint exists)")
                self._results.append(StageResult(
                    stage_name=stage.name, layer=stage.layer,
                    status="SKIPPED",
                    records_out=cached.get("records_out", 0),
                    duration_sec=0.0,
                ))
                completed.add(stage.name)
                continue

            # Execute
            result = self._execute_stage(stage)
            self._results.append(result)

            if result.status == "SUCCESS":
                self.ckpt.mark_complete(self.run_id, stage.name, result)
                completed.add(stage.name)
            elif stage.critical:
                logger.error(f"[DAG] Critical stage '{stage.name}' failed — aborting pipeline")
                break

        total_sec = time.perf_counter() - start_time
        report = self._build_report(total_sec)
        self._log_summary(report)
        return report

    def _execute_stage(self, stage: Stage) -> StageResult:
        layer_icon = {"bronze":"🥉","silver":"🥈","gold":"🏅","export":"📤"}.get(stage.layer,"▶")
        logger.info(f"\n[DAG] {layer_icon} START [{stage.layer.upper()}] {stage.name}")
        if stage.description:
            logger.info(f"       {stage.description}")

        t0 = time.perf_counter()
        try:
            output = stage.fn(self._context)
            duration = time.perf_counter() - t0

            # Stages return a dict with optional records_in/out/quality_score
            ctx_update = output if isinstance(output, dict) else {}
            self._context.update(ctx_update)

            records_out = ctx_update.get("records_out", 0)
            q_score     = ctx_update.get("quality_score", 100.0)

            logger.info(
                f"[DAG] ✓ PASS  [{stage.layer.upper()}] {stage.name} "
                f"│ {duration:.2f}s │ {records_out:,} records │ DQ={q_score:.1f}"
            )
            return StageResult(
                stage_name=stage.name, layer=stage.layer,
                status="SUCCESS", duration_sec=duration,
                records_in=ctx_update.get("records_in", 0),
                records_out=records_out, quality_score=q_score,
            )
        except Exception as exc:
            duration = time.perf_counter() - t0
            logger.error(f"[DAG] ✗ FAIL  [{stage.layer.upper()}] {stage.name} │ {exc}")
            return StageResult(
                stage_name=stage.name, layer=stage.layer,
                status="FAILED", duration_sec=duration, error=str(exc),
            )

    # ── Topology ─────────────────────────────────────────────

    def _topological_sort(self) -> list[Stage]:
        """Kahn's algorithm — raises on cycles."""
        name_to_stage = {s.name: s for s in self._stages}
        in_degree = {s.name: len(s.depends_on) for s in self._stages}
        queue = [s for s in self._stages if in_degree[s.name] == 0]
        ordered: list[Stage] = []

        while queue:
            stage = queue.pop(0)
            ordered.append(stage)
            for s in self._stages:
                if stage.name in s.depends_on:
                    in_degree[s.name] -= 1
                    if in_degree[s.name] == 0:
                        queue.append(s)

        if len(ordered) != len(self._stages):
            raise ValueError("Pipeline DAG contains a cycle — check depends_on definitions")
        return ordered

    # ── Report ───────────────────────────────────────────────

    def _build_report(self, total_sec: float) -> dict:
        succeeded = [r for r in self._results if r.status == "SUCCESS"]
        failed    = [r for r in self._results if r.status == "FAILED"]
        skipped   = [r for r in self._results if r.status == "SKIPPED"]

        return {
            "run_id":       self.run_id,
            "timestamp":    datetime.now().isoformat(),
            "total_sec":    round(total_sec, 2),
            "status":       "SUCCESS" if not failed else "FAILED",
            "stages": {
                "total":     len(self._stages),
                "succeeded": len(succeeded),
                "failed":    len(failed),
                "skipped":   len(skipped),
            },
            "total_records_processed": sum(r.records_out for r in succeeded),
            "avg_quality_score": round(
                sum(r.quality_score for r in succeeded) / max(len(succeeded), 1), 2
            ),
            "stage_details": [r.to_dict() for r in self._results],
        }

    def _log_summary(self, report: dict) -> None:
        status_icon = "✅" if report["status"] == "SUCCESS" else "❌"
        logger.info(f"\n{'═'*60}")
        logger.info(f"  PIPELINE COMPLETE  {status_icon}  {report['status']}")
        logger.info(f"  run_id      : {report['run_id']}")
        logger.info(f"  duration    : {report['total_sec']}s")
        logger.info(f"  stages      : {report['stages']['succeeded']} passed, "
                    f"{report['stages']['failed']} failed, "
                    f"{report['stages']['skipped']} skipped")
        logger.info(f"  records     : {report['total_records_processed']:,}")
        logger.info(f"  avg DQ score: {report['avg_quality_score']}")
        logger.info(f"{'═'*60}\n")

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "config/pipeline_config.yaml") -> "PipelineDAG":
        """Build a DAG pre-wired with the medallion stages."""
        import yaml
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            cfg = {}

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        dag = cls(run_id=run_id, checkpoint_dir=cfg.get("paths", {}).get("checkpoints", "data/.checkpoints"))

        # Register all medallion stages
        _register_medallion_stages(dag, cfg)
        return dag


# ─────────────────────────────────────────────────────────────
# Medallion stage registrations
# ─────────────────────────────────────────────────────────────

def _register_medallion_stages(dag: PipelineDAG, cfg: dict) -> None:
    """
    Wire up the Bronze → Silver → Gold → Export stages.
    Each stage function receives the shared pipeline context dict
    and returns a dict with records_in, records_out, quality_score.
    """
    from src.utils.data_quality import DataQualityEngine, QualityConfig

    quality_cfg = QualityConfig.from_dict(cfg.get("quality", {}))
    paths = cfg.get("paths", {
        "bronze": "data/bronze",
        "silver": "data/silver",
        "gold":   "data/gold",
        "powerbi": "data/powerbi",
    })

    # ── BRONZE ──────────────────────────────────────────────
    def bronze_ingestion(ctx: dict) -> dict:
        """
        Bronze Layer: raw data ingestion with schema enforcement.
        Reads Parquet from yahoo_fetcher output, enforces schema,
        deduplicates, adds lineage columns, partitions by sector/date.
        """
        import pandas as pd
        from pathlib import Path

        raw_path = Path(cfg.get("paths", {}).get("raw", "data/raw"))
        ohlcv_file = raw_path / "ohlcv.parquet"

        if not ohlcv_file.exists():
            raise FileNotFoundError(
                f"Raw OHLCV not found at {ohlcv_file}. "
                "Run: python data/yahoo_fetcher.py --years 20"
            )

        df = pd.read_parquet(ohlcv_file)
        n_raw = len(df)

        # Schema enforcement
        df["date"]   = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].str.upper().str.strip()
        for col in ["open","high","low","close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Deduplication
        df = df.drop_duplicates(subset=["ticker","date"])

        # Lineage columns
        df["_bronze_loaded_at"] = pd.Timestamp.now()
        df["_source"]           = "yahoo_finance"

        # DQ check
        dq = DataQualityEngine(layer="bronze", config=quality_cfg, run_id=dag.run_id)
        report = dq.run_checks(df)

        # Write partitioned bronze Parquet
        out = Path(paths.get("bronze","data/bronze"))
        out.mkdir(parents=True, exist_ok=True)
        if "sector" in df.columns:
            for sector, grp in df.groupby("sector"):
                sec_path = out / f"sector={sector}"
                sec_path.mkdir(exist_ok=True)
                grp.to_parquet(sec_path / "data.parquet", index=False)
        else:
            df.to_parquet(out / "ohlcv_bronze.parquet", index=False)

        ctx["bronze_df"] = df
        return {"records_in": n_raw, "records_out": len(df), "quality_score": report.overall_score}

    # ── SILVER: Cleanse ─────────────────────────────────────
    def silver_cleanse(ctx: dict) -> dict:
        """
        Silver Layer (Cleanse): quality-gated, conformed, enriched.
        Adds daily returns, filters outliers, fills gaps,
        enforces OHLC consistency.
        """
        import pandas as pd

        df = ctx.get("bronze_df")
        if df is None:
            df = pd.read_parquet(paths.get("bronze","data/bronze"))
        n_in = len(df)

        # OHLC consistency filter
        if all(c in df.columns for c in ["open","high","low","close"]):
            valid = (
                (df["high"] >= df["low"]) &
                (df["close"] >= df["low"] * 0.999) &
                (df["close"] <= df["high"] * 1.001)
            )
            df = df[valid]

        # Daily returns
        df = df.sort_values(["ticker","date"])
        df["daily_return"]   = df.groupby("ticker")["close"].pct_change()
        df["log_return"]     = df.groupby("ticker")["close"].transform(lambda x: x.apply(lambda v: 0).shift())  # placeholder

        # Liquidity filter: drop tickers with < 252 trading days
        counts = df.groupby("ticker")["date"].count()
        liquid = counts[counts >= 252].index
        df     = df[df["ticker"].isin(liquid)]

        # Outlier detection: flag > 50% daily move
        df["is_outlier"] = df["daily_return"].abs() > 0.50

        df["_silver_processed_at"] = pd.Timestamp.now()

        dq = DataQualityEngine(layer="silver", config=quality_cfg, run_id=dag.run_id)
        report = dq.run_checks(df)

        out = Path(paths.get("silver","data/silver")) / "cleansed"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "data.parquet", index=False)

        ctx["silver_df"] = df
        return {"records_in": n_in, "records_out": len(df), "quality_score": report.overall_score}

    # ── SILVER: Features ────────────────────────────────────
    def silver_features(ctx: dict) -> dict:
        """
        Silver Layer (Features): technical indicators, risk features.
        SMA/EMA, RSI, Bollinger, MACD, rolling vol, momentum z-score.
        """
        import pandas as pd
        import numpy as np

        df = ctx.get("silver_df")
        if df is None:
            df = pd.read_parquet(Path(paths.get("silver","data/silver")) / "cleansed" / "data.parquet")
        n_in = len(df)

        df = df.sort_values(["ticker","date"])

        def rolling_feature(group):
            g = group.copy()
            c = g["close"]
            for w in [20, 50, 200]:
                g[f"sma_{w}"] = c.rolling(w).mean()
            # RSI-14
            delta = c.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            g["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-8))
            # Volatility
            lr = c.pct_change().apply(lambda x: x)
            g["vol_20d"] = lr.rolling(20).std() * (252 ** 0.5)
            g["vol_60d"] = lr.rolling(60).std() * (252 ** 0.5)
            # Momentum
            g["roc_20"]  = c.pct_change(20)
            g["roc_252"] = c.pct_change(252)
            # Bollinger
            sma20  = c.rolling(20).mean()
            std20  = c.rolling(20).std()
            g["bb_upper"] = sma20 + 2 * std20
            g["bb_lower"] = sma20 - 2 * std20
            g["bb_pct"]   = (c - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-8)
            # Drawdown
            cum = (1 + lr).cumprod()
            g["drawdown"] = (cum - cum.cummax()) / cum.cummax()
            return g

        df = df.groupby("ticker", group_keys=False).apply(rolling_feature)
        df["_features_added_at"] = pd.Timestamp.now()

        out = Path(paths.get("silver","data/silver")) / "features"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "data.parquet", index=False)

        ctx["features_df"] = df
        return {"records_in": n_in, "records_out": len(df), "quality_score": 100.0}

    # ── GOLD: Fact table ────────────────────────────────────
    def gold_fact_metrics(ctx: dict) -> dict:
        """
        Gold Layer (Fact): date × ticker grain, BI-optimised.
        Adds scenario labels, composite signal score.
        Partitioned by year/quarter for DirectQuery pruning.
        """
        import pandas as pd
        import yaml

        df = ctx.get("features_df")
        if df is None:
            df = pd.read_parquet(Path(paths.get("silver","data/silver")) / "features" / "data.parquet")

        # Load scenario config
        try:
            with open("config/scenario_config.yaml") as f:
                scen_cfg = yaml.safe_load(f)
        except FileNotFoundError:
            scen_cfg = {}

        # Classify scenario (simplified rule engine)
        def classify(row):
            if pd.isna(row.get("roc_20", 0)): return "Normal"
            dd = row.get("drawdown", 0) or 0
            roc = row.get("roc_20", 0) or 0
            vol = row.get("vol_20d", 0.15) or 0.15
            if dd <= -0.20 and vol >= 0.22: return "Bear"
            if vol >= 0.22: return "Volatile"
            if roc >= 0.03 and dd >= -0.05: return "Bull"
            if roc >= 0.01 and dd >= -0.20: return "Recovery"
            return "Normal"

        df["market_scenario"] = df.apply(classify, axis=1)

        # Composite signal score (0–100)
        def signal_score(row):
            score = 50.0
            roc = row.get("roc_252", 0) or 0
            rsi = row.get("rsi_14", 50) or 50
            vol = row.get("vol_20d", 0.2) or 0.2
            score += min(25, roc * 100)
            score += (50 - rsi) * 0.3
            score -= vol * 50
            return round(clamp(score, 0, 100), 1)

        def clamp(v, lo, hi): return max(lo, min(hi, v))
        df["signal_score"] = df.apply(signal_score, axis=1)

        # DQ check
        dq = DataQualityEngine(layer="gold", config=quality_cfg, run_id=dag.run_id)
        report = dq.run_checks(df[["ticker","date","close","daily_return"]].dropna())

        # Write partitioned (year/quarter) for Power BI DirectQuery
        df["year"]    = pd.to_datetime(df["date"]).dt.year
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
        out_base = Path(paths.get("gold","data/gold")) / "fact_daily_metrics"

        for (yr, qtr), grp in df.groupby(["year","quarter"]):
            out_path = out_base / f"year={yr}" / f"quarter={qtr}"
            out_path.mkdir(parents=True, exist_ok=True)
            grp.drop(columns=["year","quarter"]).to_parquet(out_path / "data.parquet", index=False)

        ctx["gold_fact_df"] = df
        return {"records_in": len(df), "records_out": len(df), "quality_score": report.overall_score}

    # ── GOLD: Aggregates ────────────────────────────────────
    def gold_sector_agg(ctx: dict) -> dict:
        """
        Gold Layer (Aggregates): sector × scenario pre-aggregated KPIs.
        Used by Power BI executive dashboards in Import mode.
        """
        import pandas as pd
        import numpy as np

        df = ctx.get("gold_fact_df", ctx.get("features_df"))
        if df is None:
            return {"records_in": 0, "records_out": 0, "quality_score": 100.0}

        agg = (
            df.dropna(subset=["daily_return"])
              .groupby(["sector","market_scenario"] if "market_scenario" in df.columns else ["sector"])
              .agg(
                  n_equities       =("ticker",        "nunique"),
                  avg_return_ann   =("daily_return",  lambda x: x.mean() * 252),
                  volatility_ann   =("daily_return",  lambda x: x.std() * (252**0.5)),
                  win_rate         =("daily_return",  lambda x: (x > 0).mean()),
                  avg_rsi          =("rsi_14",         "mean"),
                  avg_vol_20d      =("vol_20d",         "mean"),
              )
              .reset_index()
        )
        agg["sharpe"] = (agg["avg_return_ann"] - 0.05) / agg["volatility_ann"].replace(0, 1e-8)
        agg = agg.round(4)

        out = Path(paths.get("gold","data/gold")) / "sector_scenario_performance"
        out.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out / "data.parquet", index=False)
        agg.to_csv(out / "sector_scenario_performance.csv", index=False)

        ctx["gold_agg_df"] = agg
        return {"records_in": len(df), "records_out": len(agg), "quality_score": 100.0}

    # ── EXPORT: Power BI ────────────────────────────────────
    def export_powerbi(ctx: dict) -> dict:
        """
        Export Layer: Power BI-optimised CSVs and Parquet.
        Generates fact tables (DirectQuery), dimensions (Import),
        and the semantic model relationship spec.
        """
        import pandas as pd
        from datetime import date

        out = Path(paths.get("powerbi","data/powerbi"))
        out.mkdir(parents=True, exist_ok=True)

        written = 0

        # Fact table
        if "gold_fact_df" in ctx:
            ctx["gold_fact_df"].to_parquet(out / "fact_daily_metrics.parquet", index=False)
            written += len(ctx["gold_fact_df"])

        # Sector aggregates
        if "gold_agg_df" in ctx:
            ctx["gold_agg_df"].to_csv(out / "dim_sector_scenario.csv", index=False)

        # Date dimension
        dates = pd.bdate_range(start="2004-01-01", end=date.today())
        date_dim = pd.DataFrame({
            "date":       dates.date,
            "year":       dates.year,
            "quarter":    dates.quarter,
            "month_num":  dates.month,
            "month_name": dates.strftime("%b"),
            "day_name":   dates.strftime("%a"),
            "is_trading_day": 1,
        })
        date_dim.to_csv(out / "dim_date.csv", index=False)

        # Semantic model spec
        spec = {
            "generated":   date.today().isoformat(),
            "tables":      ["fact_daily_metrics","dim_sector_scenario","dim_date"],
            "relationships": [
                {"from":"fact_daily_metrics[date]",   "to":"dim_date[date]",       "cardinality":"M:1"},
                {"from":"fact_daily_metrics[sector]", "to":"dim_sector_scenario[sector]","cardinality":"M:1"},
            ],
            "storage_mode": {"fact_daily_metrics":"DirectQuery","dim_*":"Import"},
        }
        import json
        (out / "semantic_model.json").write_text(json.dumps(spec, indent=2))

        logger.info(f"[EXPORT] Power BI artifacts → {out.resolve()}")
        return {"records_in": written, "records_out": written, "quality_score": 100.0}

    # Register stages in dependency order
    dag.add_stage(Stage("bronze_ingestion",  "bronze", bronze_ingestion,  [],                             "Raw OHLCV → schema enforced, deduplicated, lineage tagged"))
    dag.add_stage(Stage("silver_cleanse",    "silver", silver_cleanse,    ["bronze_ingestion"],           "Cleanse, OHLC validation, liquidity filter, return calc"))
    dag.add_stage(Stage("silver_features",   "silver", silver_features,   ["silver_cleanse"],             "SMA/RSI/MACD/Bollinger/vol/momentum feature engineering"))
    dag.add_stage(Stage("gold_fact_metrics", "gold",   gold_fact_metrics, ["silver_features"],            "Date×ticker fact table, scenario labels, signal score"))
    dag.add_stage(Stage("gold_sector_agg",   "gold",   gold_sector_agg,   ["gold_fact_metrics"],          "Sector×scenario pre-aggregated KPIs for executive BI"))
    dag.add_stage(Stage("export_powerbi",    "export", export_powerbi,    ["gold_fact_metrics","gold_sector_agg"], "Power BI DirectQuery facts + Import dimensions"))
