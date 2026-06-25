"""
main.py
--------
Production entry point for the Equity Market Analytics Platform.

Runs the full medallion pipeline:
  Bronze (ingest) → Silver (cleanse + features) → Gold (fact + agg) → Export (Power BI)

Usage
-----
  # Full pipeline — 503 S&P 500 tickers, 20 years Yahoo Finance data
  python main.py

  # Force recompute (ignore checkpoints)
  python main.py --force

  # Quick test — 20 tickers, 2 years
  python main.py --tickers 20 --years 2

  # Custom config
  python main.py --config config/pipeline_config.yaml

  # Skip data download (use existing data/raw/)
  python main.py --skip-download

  # API only
  python main.py --api-only
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level=level,
        colorize=True,
    )
    log_path = Path("data/logs")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "pipeline_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="14 days",
        serialize=True,       # JSON — ELK/Datadog compatible
    )


def download_data(n_tickers: int, years: int) -> None:
    logger.info(f"Downloading Yahoo Finance data: {n_tickers} tickers × {years}Y")
    try:
        from data.yahoo_fetcher import download
        download(n_tickers=n_tickers, years=years)
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Data download failed: {exc}")
        sys.exit(1)


def run_pipeline(config_path: str, force: bool) -> dict:
    from src.orchestration.pipeline_dag import PipelineDAG
    dag = PipelineDAG.from_config(config_path)
    return dag.run(force=force)


def run_api() -> None:
    import uvicorn
    from api.main import app
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    logger.info("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equity Market Analytics Platform — Medallion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Full pipeline (503 tickers, 20Y)
  python main.py --force                  Force recompute all stages
  python main.py --tickers 20 --years 2  Quick test run
  python main.py --skip-download          Use existing data/raw/ data
  python main.py --api-only               Start FastAPI server only
        """
    )
    parser.add_argument("--config",        default="config/pipeline_config.yaml")
    parser.add_argument("--tickers",       type=int,  default=503)
    parser.add_argument("--years",         type=int,  default=20)
    parser.add_argument("--force",         action="store_true", help="Ignore checkpoints")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--api-only",      action="store_true", help="Start API server only")
    parser.add_argument("--log-level",     default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║   Equity Market Analytics Platform  v2.0             ║")
    logger.info("║   Bronze → Silver → Gold → Power BI                  ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    if args.api_only:
        run_api()
        return

    # Step 1: Download Yahoo Finance data
    raw_exists = (Path("data/raw/ohlcv.parquet")).exists()
    if not args.skip_download and not raw_exists:
        download_data(args.tickers, args.years)
    elif raw_exists and not args.skip_download:
        logger.info("data/raw/ohlcv.parquet found. Use --force to re-download.")

    # Step 2: Run medallion pipeline
    t0 = time.perf_counter()
    report = run_pipeline(args.config, force=args.force)
    elapsed = time.perf_counter() - t0

    # Step 3: Print summary
    status = report.get("status", "UNKNOWN")
    icon   = "✅" if status == "SUCCESS" else "❌"
    print(f"\n{icon}  Pipeline {status}  ({elapsed:.1f}s)")
    print(f"   Stages  : {report['stages']['succeeded']} passed · {report['stages']['failed']} failed · {report['stages']['skipped']} skipped")
    print(f"   Records : {report['total_records_processed']:,}")
    print(f"   DQ Score: {report['avg_quality_score']:.1f}/100")
    print(f"\n   Outputs :")
    for path in ["data/bronze", "data/silver", "data/gold", "data/powerbi"]:
        if Path(path).exists():
            print(f"     {path}/")
    print(f"\n   API     : python main.py --api-only")
    print(f"   Docs    : http://localhost:8000/docs\n")

    sys.exit(0 if status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
