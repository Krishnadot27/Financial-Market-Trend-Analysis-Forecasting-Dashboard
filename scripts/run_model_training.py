"""
scripts/run_model_training.py
──────────────────────────────
Step 3: Train and compare all models on NSE stock data.

Run from project root:
    python scripts/run_model_training.py

What it does:
  1. Loads feature-engineered data from data/processed/
  2. Temporal train/val/test split
  3. Trains LinearBaseline, RandomForest, XGBoost
  4. Walk-forward cross-validation
  5. Evaluates on held-out test set
  6. Saves best model to models/
  7. Prints comparison table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils import load_config, setup_logger, set_seed, project_path, ensure_dirs
from src.feature_engineering import FeatureEngineer
from src.data_loader import temporal_split
from src.model_training import (
    LinearBaseline, RandomForestModel, XGBoostModel,
    walk_forward_validate, compare_models,
)
from src.evaluation import ModelEvaluator
from loguru import logger


def main():
    setup_logger(log_dir="logs")
    cfg = load_config()
    set_seed(cfg["project"]["random_seed"])
    ensure_dirs(project_path("models"), project_path("results"))

    # ── Pick ticker ───────────────────────────────────────────────────────────
    ticker      = cfg["data"]["tickers"][0]       # RELIANCE.NS
    safe_ticker = ticker.replace(".", "_")
    feat_path   = project_path("data", "processed", f"{safe_ticker}_features.csv")

    if not feat_path.exists():
        logger.error(f"Feature file not found: {feat_path}")
        logger.error("Run Step 2 first: python scripts/run_feature_engineering.py")
        return

    logger.info("=" * 60)
    logger.info(f"  STEP 3 — Model Training: {ticker}")
    logger.info("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────────
    df_feat = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
    logger.info(f"Loaded: {df_feat.shape[0]} rows × {df_feat.shape[1]} columns")

    # ── Choose task and target ────────────────────────────────────────────────
    TASK   = cfg["model"]["task"]           # 'regression' or 'classification'
    TARGET = "Target_Return" if TASK == "regression" else "Target_Dir"
    logger.info(f"Task: {TASK}  |  Target: {TARGET}")

    # ── X, y ─────────────────────────────────────────────────────────────────
    X, y = FeatureEngineer.get_X_y(df_feat, target=TARGET)
    logger.info(f"Features: {X.shape[1]}  |  Samples: {len(X)}")

    # ── Temporal split ────────────────────────────────────────────────────────
    n         = len(X)
    n_test    = int(n * cfg["split"]["test_ratio"])
    n_val     = int(n * cfg["split"]["val_ratio"])
    n_train   = n - n_test - n_val

    X_train = X.iloc[:n_train];       y_train = y.iloc[:n_train]
    X_val   = X.iloc[n_train:n_train+n_val]; y_val = y.iloc[n_train:n_train+n_val]
    X_test  = X.iloc[n_train+n_val:]; y_test  = y.iloc[n_train+n_val:]

    logger.info(
        f"Split → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}"
    )

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    logger.info("\n── Walk-Forward Cross Validation ──")
    cv_results = walk_forward_validate(
        XGBoostModel,
        X_train, y_train,
        n_splits=cfg["split"]["n_cv_splits"],
        task=TASK,
        model_kwargs={"n_estimators": 200, "learning_rate": 0.05},
    )

    # ── Model comparison ──────────────────────────────────────────────────────
    logger.info("\n── Model Comparison (Test Set) ──")
    comparison = compare_models(X_train, y_train, X_test, y_test, task=TASK)

    # ── Train final XGBoost on Train+Val ─────────────────────────────────────
    logger.info("\n── Training Final XGBoost on Train+Val ──")
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    final_model = XGBoostModel(
        task=TASK,
        **cfg["model"]["xgboost"],
    )
    final_model.fit(X_trainval, y_trainval, X_test, y_test)

    # ── Final evaluation ──────────────────────────────────────────────────────
    ev     = ModelEvaluator(task=TASK)
    y_pred = final_model.predict(X_test)

    logger.info("\n── Final Test Set Evaluation ──")
    if TASK == "regression":
        metrics = ev.regression_report(y_test, y_pred, label=f"XGBoost — {ticker}")
    else:
        y_proba = final_model.predict_proba(X_test)
        metrics = ev.classification_report(y_test, y_pred, y_proba, label=f"XGBoost — {ticker}")

    # ── Feature importance ────────────────────────────────────────────────────
    logger.info("\n── Top 15 Feature Importances ──")
    imp = final_model.feature_importance(top_n=15)
    print(imp.to_string())

    # ── Save model ────────────────────────────────────────────────────────────
    save_path = project_path("models", f"xgboost_{TASK}_{safe_ticker}.pkl")
    final_model.save(save_path)

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = project_path("results", f"model_comparison_{safe_ticker}.csv")
    comparison.to_csv(results_path)
    logger.success(f"Results saved → {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("  Training complete!")
    logger.info(f"  ✔  Model saved → models/xgboost_{TASK}_{safe_ticker}.pkl")
    logger.info(f"  ✔  Comparison  → results/model_comparison_{safe_ticker}.csv")
    logger.info("  Next step → python scripts/run_backtesting.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()