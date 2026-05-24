"""
scripts/run_training_v2.py
───────────────────────────
Upgraded training script using the v2 pipeline:
  - Meaningful target (filters noise < 0.3%)
  - Strict temporal splits
  - SHAP/built-in feature selection (125 → 40 features)
  - Optuna Bayesian tuning
  - Ensemble (XGBoost + RF + LightGBM)
  - Walk-forward cross-validation

Run on Kaggle (paste as notebook cell):

    !pip install -q loguru xgboost==2.0.3 lightgbm optuna yfinance ta pyyaml scikit-learn joblib shap

    import sys, shutil, os
    dst = '/kaggle/working/project'
    if not os.path.exists(dst):
        src = '/kaggle/input/datasets/krishnamali27/nse-ml-project/predictive analysis pr'
        shutil.copytree(src, dst)
    if dst not in sys.path: sys.path.insert(0, dst)
    os.chdir(dst)
    for k in [k for k in sys.modules if k.startswith('src')]:
        del sys.modules[k]

    exec(open('scripts/run_training_v2.py').read())

Locally:
    python scripts/run_training_v2.py
    python scripts/run_training_v2.py --notune    # skip Optuna (faster)
    python scripts/run_training_v2.py --ticker RELIANCE.NS  # single ticker
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf
from loguru import logger

from src.utils import load_config, setup_logger, project_path
from src.data_loader import NSEDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_v2 import train_v2


def run_v2_training(
    tickers:  list[str],
    task:     str  = "classification",
    tune:     bool = True,
    n_trials: int  = 40,
    top_feat: int  = 40,
    threshold: float = 0.003,
) -> pd.DataFrame:
    """
    Train v2 ensemble model for all tickers.
    Returns a summary DataFrame of results.
    """
    setup_logger(log_dir="logs")
    cfg     = load_config()
    results = []

    logger.info("=" * 65)
    logger.info(f"  NSE Alpha v2 Training  [{task.upper()}]")
    logger.info(f"  Tickers: {len(tickers)}")
    logger.info(f"  Optuna:  {'Yes (' + str(n_trials) + ' trials)' if tune else 'No (defaults)'}")
    logger.info(f"  Features: top {top_feat} (from ~125)")
    logger.info(f"  Target threshold: ±{threshold*100:.1f}% moves only")
    logger.info("=" * 65)

    start_date = cfg["data"].get("start_date", "2015-01-01")
    end_date   = cfg["data"].get("end_date",   "2025-12-31")

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"\n[{i}/{len(tickers)}] {ticker}")
        try:
            # Load data
            loader = NSEDataLoader(ticker=ticker, start=start_date, end=end_date)
            df_raw = loader.load()

            if df_raw is None or len(df_raw) < 400:
                logger.warning(f"  Not enough data for {ticker} ({len(df_raw) if df_raw is not None else 0} rows)")
                results.append({"ticker": ticker, "status": "skipped", "reason": "insufficient_data"})
                continue

            # Build features
            fe      = FeatureEngineer(df_raw, config=cfg)
            df_feat = fe.build()

            if len(df_feat) < 300:
                logger.warning(f"  Too few rows after feature build: {len(df_feat)}")
                results.append({"ticker": ticker, "status": "skipped", "reason": "too_few_rows"})
                continue

            # Train v2
            result = train_v2(
                ticker        = ticker,
                df_feat       = df_feat,
                task          = task,
                tune          = tune,
                n_trials      = n_trials,
                top_features  = top_feat,
                threshold     = threshold,
                save          = True,
            )

            if result is None:
                results.append({"ticker": ticker, "status": "failed"})
                continue

            row = {
                "ticker": ticker,
                "status": "ok",
                "rows":   len(df_feat),
                "features": len(result.features),
                "train_time": round(result.train_time, 1),
                **result.metrics,
            }
            results.append(row)

            # Progress log
            if task == "classification":
                logger.success(
                    f"  ✔ {ticker}  "
                    f"CV_Acc={result.metrics.get('accuracy','?'):.4f}  "
                    f"Test_Acc={result.metrics.get('test_accuracy','?'):.4f}  "
                    f"AUC={result.metrics.get('test_auc','?'):.4f}  "
                    f"time={result.train_time:.0f}s"
                )
            else:
                logger.success(
                    f"  ✔ {ticker}  "
                    f"CV_RMSE={result.metrics.get('rmse','?')}  "
                    f"Test_R²={result.metrics.get('test_r2','?')}  "
                    f"time={result.train_time:.0f}s"
                )

        except Exception as e:
            logger.error(f"  ✘ {ticker}: {e}")
            results.append({"ticker": ticker, "status": "error", "reason": str(e)[:80]})

    # Summary
    df_results = pd.DataFrame(results)
    ok         = df_results[df_results["status"] == "ok"]

    logger.info("\n" + "=" * 65)
    logger.info(f"  TRAINING COMPLETE")
    logger.info(f"  Successful: {len(ok)}/{len(tickers)}")

    if len(ok) > 0 and task == "classification":
        mean_acc = ok["test_accuracy"].mean() if "test_accuracy" in ok.columns else 0
        mean_auc = ok["test_auc"].mean()      if "test_auc"      in ok.columns else 0
        logger.info(f"  Mean Test Accuracy: {mean_acc:.4f}")
        logger.info(f"  Mean Test AUC:      {mean_auc:.4f}")
    elif len(ok) > 0:
        mean_r2 = ok["test_r2"].mean() if "test_r2" in ok.columns else 0
        logger.info(f"  Mean Test R²: {mean_r2:.4f}")

    logger.info("=" * 65)

    # Save results CSV
    out_path = project_path("results", f"v2_training_{task}.csv")
    df_results.to_csv(out_path, index=False)
    logger.info(f"Results saved → {out_path}")

    return df_results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    default="classification",
                        choices=["classification", "regression"])
    parser.add_argument("--notune",  action="store_true",
                        help="Skip Optuna tuning (use defaults, faster)")
    parser.add_argument("--trials",  type=int, default=40,
                        help="Number of Optuna trials per ticker")
    parser.add_argument("--features",type=int, default=40,
                        help="Number of top features to select")
    parser.add_argument("--threshold",type=float, default=0.003,
                        help="Min price move % for meaningful target")
    parser.add_argument("--ticker",  type=str, default=None,
                        help="Train single ticker only")
    args = parser.parse_args()

    cfg     = load_config()
    tickers = [args.ticker] if args.ticker else cfg["data"]["tickers"]

    run_v2_training(
        tickers   = tickers,
        task      = args.task,
        tune      = not args.notune,
        n_trials  = args.trials,
        top_feat  = args.features,
        threshold = args.threshold,
    )


if __name__ == "__main__":
    main()