"""
scripts/run_backtesting.py
───────────────────────────
Step 4: Run backtesting on trained model predictions.

Run from project root:
    python scripts/run_backtesting.py

What it does:
  1. Loads feature data from data/processed/
  2. Loads saved XGBoost model from models/
  3. Generates predictions on test set
  4. Runs backtest with realistic NSE transaction costs
  5. Prints full performance report vs Buy-and-Hold
  6. Saves interactive charts to results/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils import load_config, setup_logger, set_seed, project_path, ensure_dirs
from src.feature_engineering import FeatureEngineer
from src.data_loader import temporal_split
from src.model_training import XGBoostModel
from src.backtesting import Backtester, run_backtest
from src.evaluation import ModelEvaluator
from loguru import logger


def main():
    setup_logger(log_dir="logs")
    cfg = load_config()
    set_seed(cfg["project"]["random_seed"])
    ensure_dirs(project_path("results"))

    ticker      = cfg["data"]["tickers"][0]
    safe_ticker = ticker.replace(".", "_")
    TASK        = cfg["model"]["task"]
    TARGET      = "Target_Return" if TASK == "regression" else "Target_Dir"

    logger.info("=" * 60)
    logger.info(f"  STEP 4 — Backtesting: {ticker}")
    logger.info("=" * 60)

    # ── Load feature data ─────────────────────────────────────────────────────
    feat_path = project_path("data", "processed", f"{safe_ticker}_features.csv")
    if not feat_path.exists():
        logger.error("Features not found. Run Step 2 first.")
        return

    df_feat = pd.read_csv(feat_path, index_col="Date", parse_dates=True)

    # ── Load trained model ────────────────────────────────────────────────────
    model_path = project_path("models", f"xgboost_{TASK}_{safe_ticker}.pkl")
    if not model_path.exists():
        logger.error("Model not found. Run Step 3 first.")
        return

    model = XGBoostModel.load(model_path)
    logger.info(f"Model loaded: {model_path.name}")

    # ── Use only the TEST split for backtesting ───────────────────────────────
    n       = len(df_feat)
    n_test  = int(n * cfg["split"]["test_ratio"])
    df_test = df_feat.iloc[n - n_test:]
    logger.info(
        f"Test period: {df_test.index[0].date()} → "
        f"{df_test.index[-1].date()}  ({len(df_test)} days)"
    )

    # ── Run backtest ──────────────────────────────────────────────────────────
    output = run_backtest(
        df_features      = df_test,
        model            = model,
        target_col       = TARGET,
        price_col        = "Close",
        transaction_cost = cfg["backtest"]["transaction_cost"],
        slippage         = cfg["backtest"]["slippage"],
        ticker           = ticker,
        save_dir         = project_path("results"),
    )

    results = output["results"]
    metrics = output["metrics"]

    # ── Monthly returns table ─────────────────────────────────────────────────
    logger.info("\n── Monthly Returns ──")
    bt = output["backtester"]
    bt.monthly_returns_heatmap(
        save_path=project_path("results", f"monthly_heatmap_{safe_ticker}.html")
    )

    # ── Direction accuracy (if regression model) ──────────────────────────────
    if TASK == "regression":
        actual_dir   = (results["bh_return"] > 0).astype(int)
        pred_dir     = (results["signal"] > 0).astype(int)
        dir_accuracy = (actual_dir == pred_dir).mean() * 100
        logger.info(f"\nDirection Accuracy: {dir_accuracy:.1f}%")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Backtesting complete!")
    logger.info(f"  ✔  Charts saved → results/")
    logger.info(f"  ✔  Alpha vs B&H: {metrics['alpha']:+.2f}%")
    sharpe = metrics["strategy"]["Sharpe Ratio"]
    logger.info(f"  ✔  Strategy Sharpe: {sharpe:.3f}"
                + ("  (Good!)" if sharpe > 1.0 else "  (Needs improvement)"))
    logger.info("=" * 60)
    logger.info("  Project complete! Next → build the Streamlit dashboard")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()