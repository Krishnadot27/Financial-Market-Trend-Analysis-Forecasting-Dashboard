"""
scripts/run_data_pipeline.py
─────────────────────────────
End-to-end demo of Step 1: Data ingestion pipeline.

Run from the project root:
    python scripts/run_data_pipeline.py

What it does:
  1. Loads config.yaml
  2. Downloads OHLCV data for all NSE tickers + NIFTY 50 benchmark
  3. Validates data quality
  4. Performs temporal train/val/test split
  5. Saves processed DataFrames to data/processed/
  6. Prints a summary report
"""

import sys
from pathlib import Path

# Make sure src/ is on path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logger, set_seed, project_path, ensure_dirs
from src.data_loader import download_all_tickers, load_ticker_from_config, temporal_split
from loguru import logger
import pandas as pd


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    setup_logger(log_dir="logs")
    cfg  = load_config()
    set_seed(cfg["project"]["random_seed"])

    ensure_dirs(
        project_path("data", "raw"),
        project_path("data", "processed"),
        project_path("logs"),
    )

    logger.info("=" * 60)
    logger.info("  STEP 1 — Data Pipeline")
    logger.info("=" * 60)

    # ── 1. Download all tickers ────────────────────────────────────────────────
    tickers = cfg["data"]["tickers"]
    logger.info(f"Tickers to download: {tickers}")

    dfs = download_all_tickers(
        tickers=tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        interval=cfg["data"]["interval"],
    )

    if not dfs:
        logger.error("No data downloaded. Check your internet connection.")
        return

    # ── 2. Download benchmark (NIFTY 50) ──────────────────────────────────────
    benchmark_ticker = cfg["data"]["benchmark"]
    logger.info(f"Downloading benchmark: {benchmark_ticker}")
    try:
        benchmark_dfs = download_all_tickers(
            tickers=[benchmark_ticker],
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
        )
        benchmark_df = benchmark_dfs.get(benchmark_ticker)
        if benchmark_df is not None:
            bench_path = project_path("data", "processed", "NIFTY50_benchmark.csv")
            benchmark_df.to_csv(bench_path)
            logger.success(f"Benchmark saved → {bench_path.name}")
    except Exception as e:
        logger.warning(f"Benchmark download failed: {e}")

    # ── 3. Quality summary ─────────────────────────────────────────────────────
    logger.info("\n── Data Quality Summary ──")
    summary_rows = []
    for ticker, df in dfs.items():
        summary_rows.append({
            "Ticker":      ticker,
            "Rows":        len(df),
            "Start":       str(df.index[0].date()),
            "End":         str(df.index[-1].date()),
            "Nulls":       df.isnull().sum().sum(),
            "Avg Volume":  f"{df['Volume'].mean():,.0f}",
            "Min Close":   f"₹{df['Close'].min():,.2f}",
            "Max Close":   f"₹{df['Close'].max():,.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    # ── 4. Train/Val/Test split (demonstrate on one ticker) ───────────────────
    sample_ticker = tickers[0]
    sample_df = dfs[sample_ticker]

    logger.info(f"\n── Temporal Split Demo: {sample_ticker} ──")
    train, val, test = temporal_split(
        sample_df,
        test_ratio=cfg["split"]["test_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
    )

    # ── 5. Save processed splits ───────────────────────────────────────────────
    processed_dir = project_path("data", "processed")
    for ticker, df in dfs.items():
        safe = ticker.replace(".", "_").replace("^", "IDX_")
        out_path = processed_dir / f"{safe}_clean.csv"
        df.to_csv(out_path)

    logger.success(f"Saved {len(dfs)} clean CSVs → data/processed/")

    # ── 6. Final report ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Pipeline complete!")
    logger.info(f"  ✔  {len(dfs)} tickers downloaded and validated")
    logger.info(f"  ✔  Data range: {cfg['data']['start_date']} → {cfg['data']['end_date']}")
    logger.info(f"  ✔  Split: {100*(1-cfg['split']['test_ratio']-cfg['split']['val_ratio']):.0f}% train / "
                f"{100*cfg['split']['val_ratio']:.0f}% val / "
                f"{100*cfg['split']['test_ratio']:.0f}% test")
    logger.info(f"  ✔  Cached CSVs in: data/raw/")
    logger.info(f"  ✔  Clean CSVs in:  data/processed/")
    logger.info("=" * 60)
    logger.info("  Next step → run: python scripts/run_feature_engineering.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()