"""
scripts/run_feature_engineering.py
────────────────────────────────────
Step 2: Run feature engineering on all downloaded NSE tickers.

Run from project root:
    python scripts/run_feature_engineering.py

What it does:
  1. Loads clean OHLCV from data/processed/
  2. Runs FeatureEngineer.build() on each ticker
  3. Saves feature matrices to data/processed/<TICKER>_features.csv
  4. Prints feature count, shape, and class balance
  5. Saves a feature correlation heatmap to results/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils import load_config, setup_logger, set_seed, project_path, ensure_dirs
from src.feature_engineering import FeatureEngineer
from loguru import logger


def main():
    setup_logger(log_dir="logs")
    cfg = load_config()
    set_seed(cfg["project"]["random_seed"])

    processed_dir = project_path("data", "processed")
    results_dir   = project_path("results")
    ensure_dirs(results_dir)

    logger.info("=" * 60)
    logger.info("  STEP 2 — Feature Engineering")
    logger.info("=" * 60)

    tickers = cfg["data"]["tickers"]
    summary_rows = []

    for ticker in tickers:
        safe = ticker.replace(".", "_").replace("^", "IDX_")
        csv_path = processed_dir / f"{safe}_clean.csv"

        if not csv_path.exists():
            logger.warning(f"Clean CSV not found for {ticker} — run Step 1 first")
            continue

        # ── Load raw OHLCV ────────────────────────────────────────────────────
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        logger.info(f"\nProcessing {ticker}  ({len(df)} rows)")

        # ── Build features ────────────────────────────────────────────────────
        fe      = FeatureEngineer(df, config=cfg)
        df_feat = fe.build()

        # ── Get X, y ──────────────────────────────────────────────────────────
        X, y_dir = FeatureEngineer.get_X_y(df_feat, target="Target_Dir")
        _, y_ret = FeatureEngineer.get_X_y(df_feat, target="Target_Return")

        # ── Class balance ─────────────────────────────────────────────────────
        up_pct = y_dir.mean() * 100
        logger.info(f"  Features : {X.shape[1]}")
        logger.info(f"  Rows     : {X.shape[0]}")
        logger.info(f"  Up days  : {up_pct:.1f}%  |  Down days: {100-up_pct:.1f}%")

        # ── Save feature CSV ──────────────────────────────────────────────────
        out_path = processed_dir / f"{safe}_features.csv"
        df_feat.to_csv(out_path)
        logger.success(f"  Saved → {out_path.name}")

        summary_rows.append({
            "Ticker":   ticker,
            "Rows":     X.shape[0],
            "Features": X.shape[1],
            "Up%":      f"{up_pct:.1f}",
            "NaN cols": X.isnull().any().sum(),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    if summary_rows:
        print("\n" + pd.DataFrame(summary_rows).to_string(index=False))

    # ── Feature list (from first ticker) ──────────────────────────────────────
    if summary_rows:
        first_ticker = tickers[0].replace(".", "_")
        feat_csv     = processed_dir / f"{first_ticker}_features.csv"
        if feat_csv.exists():
            df_sample    = pd.read_csv(feat_csv, index_col="Date", parse_dates=True)
            feature_cols = FeatureEngineer.get_feature_cols(df_sample)
            print(f"\nFeature list ({len(feature_cols)} total):")
            for i, f in enumerate(sorted(feature_cols), 1):
                print(f"  {i:>3}. {f}")

    logger.info("\n" + "=" * 60)
    logger.info("  Feature engineering complete!")
    logger.info("  Next step → python scripts/run_model_training.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()