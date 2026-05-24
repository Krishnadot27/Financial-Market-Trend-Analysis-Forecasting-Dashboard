"""
src/pipeline.py
────────────────
Master pipeline — runs the entire project end-to-end in one call.

Usage:
    from src.pipeline import run_full_pipeline
    run_full_pipeline(tickers=["RELIANCE.NS", "TCS.NS"])

Or from command line:
    python src/pipeline.py

Steps executed:
  1. Download & validate OHLCV data
  2. Feature engineering (125 features)
  3. Train XGBoost model per ticker
  4. Backtest per ticker
  5. Build multi-ticker portfolio
  6. Generate final report
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.utils import load_config, setup_logger, set_seed, project_path, ensure_dirs
from src.data_loader import NSEDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import XGBoostModel, compare_models
from src.evaluation import ModelEvaluator
from src.backtesting import Backtester
from src.portfolio import Portfolio


def run_full_pipeline(
    tickers:    Optional[list[str]] = None,
    task:       str = "regression",
    save_plots: bool = True,
) -> dict:
    """
    Run the complete NSE ML pipeline end-to-end.

    Parameters
    ----------
    tickers    : List of NSE tickers (defaults to config.yaml list)
    task       : 'regression' | 'classification'
    save_plots : Save HTML charts to results/

    Returns
    -------
    dict with keys: models, backtest_results, portfolio_results, summary
    """
    t_start = time.perf_counter()

    setup_logger(log_dir="logs")
    cfg = load_config()
    set_seed(cfg["project"]["random_seed"])
    ensure_dirs(
        project_path("data", "raw"),
        project_path("data", "processed"),
        project_path("models"),
        project_path("results"),
    )

    tickers = tickers or cfg["data"]["tickers"]
    TARGET  = "Target_Return" if task == "regression" else "Target_Dir"

    logger.info("=" * 65)
    logger.info("  NSE STOCK PREDICTION — FULL PIPELINE")
    logger.info(f"  Tickers: {tickers}")
    logger.info(f"  Task:    {task}")
    logger.info("=" * 65)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Data
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── STEP 1: Data Collection ──")
    from datetime import datetime as _dt
    start_date = cfg["data"].get("start_date", "2015-01-01")
    end_date   = cfg["data"].get("end_date", _dt.now().strftime("%Y-%m-%d"))
    prices_raw = {}

    for ticker in tickers:
        try:
            loader = NSEDataLoader(
                ticker=ticker,
                start=start_date,
                end=end_date,
            )
            df = loader.load()
            if df is not None and len(df) >= 200:
                prices_raw[ticker] = df
                logger.info(f"  ✔  {ticker}: {len(df)} rows")
            else:
                logger.warning(f"  ✘  {ticker}: insufficient data, skipping")
        except Exception as e:
            logger.warning(f"  ✘  {ticker}: {e}, skipping")

    if not prices_raw:
        raise RuntimeError("No valid data downloaded. Check internet connection.")

    valid_tickers = list(prices_raw.keys())

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── STEP 2: Feature Engineering ──")
    features_dict = {}

    for ticker, df_raw in prices_raw.items():
        fe      = FeatureEngineer(df_raw, config=cfg)
        df_feat = fe.build()
        features_dict[ticker] = df_feat

        feat_cols = FeatureEngineer.get_feature_cols(df_feat)
        logger.info(
            f"  ✔  {ticker}: {len(feat_cols)} features × {len(df_feat)} rows"
        )

        # Save processed features
        safe = ticker.replace(".", "_")
        out  = project_path("data", "processed", f"{safe}_features.csv")
        df_feat.to_csv(out)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Model Training
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── STEP 3: Model Training ──")
    models_dict  = {}
    metrics_dict = {}
    ev           = ModelEvaluator(task=task)

    for ticker in valid_tickers:
        df_feat = features_dict[ticker]
        X, y    = FeatureEngineer.get_X_y(df_feat, target=TARGET)

        # Temporal split
        n       = len(X)
        n_test  = int(n * cfg["split"]["test_ratio"])
        n_val   = int(n * cfg["split"]["val_ratio"])
        n_train = n - n_test - n_val

        X_tr = X.iloc[:n_train];           y_tr = y.iloc[:n_train]
        X_v  = X.iloc[n_train:n_train+n_val]; y_v = y.iloc[n_train:n_train+n_val]
        X_te = X.iloc[n_train+n_val:];     y_te = y.iloc[n_train+n_val:]

        # Train on train+val, evaluate on test
        X_tv = pd.concat([X_tr, X_v]); y_tv = pd.concat([y_tr, y_v])

        model = XGBoostModel(task=task, **cfg["model"]["xgboost"])
        model.fit(X_tv, y_tv, X_te, y_te)

        # Evaluate
        y_pred = model.predict(X_te)
        if task == "regression":
            m = ev.regression_report(y_te, y_pred, label=ticker)
        else:
            m = ev.classification_report(y_te, y_pred, label=ticker)

        # Save model
        safe       = ticker.replace(".", "_")
        model_path = project_path("models", f"xgboost_{task}_{safe}.pkl")
        model.save(model_path)

        models_dict[ticker]  = model
        metrics_dict[ticker] = m
        logger.info(f"  ✔  {ticker}: model saved")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Per-Ticker Backtesting
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── STEP 4: Per-Ticker Backtesting ──")
    backtest_dict   = {}
    signals_dict    = {}
    prices_dict     = {}

    for ticker in valid_tickers:
        df_feat      = features_dict[ticker]
        model        = models_dict[ticker]
        feature_cols = FeatureEngineer.get_feature_cols(df_feat)
        X            = df_feat[feature_cols]
        prices       = df_feat["Close"]

        signals = pd.Series(model.predict(X), index=X.index)

        bt      = Backtester(
            prices, signals,
            transaction_cost=cfg["backtest"]["transaction_cost"],
            slippage=cfg["backtest"]["slippage"],
        )
        results = bt.run()
        metrics = bt.report()

        if save_plots:
            safe = ticker.replace(".", "_")
            bt.plot(
                ticker=ticker,
                save_path=project_path("results", f"backtest_{safe}.html"),
            )

        backtest_dict[ticker] = {"results": results, "metrics": metrics}
        signals_dict[ticker]  = signals
        prices_dict[ticker]   = prices
        logger.info(
            f"  ✔  {ticker}: "
            f"Sharpe={metrics['strategy']['Sharpe Ratio']:.3f}  "
            f"Alpha={metrics['alpha']:+.1f}%"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Portfolio
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── STEP 5: Multi-Ticker Portfolio ──")
    portfolio_results = {}

    for strategy in ["equal_weight", "signal_weight", "momentum_weight"]:
        pf      = Portfolio(
            signals_dict=signals_dict,
            prices_dict=prices_dict,
            strategy=strategy,
            rebalance_freq="W",
            transaction_cost=cfg["backtest"]["transaction_cost"],
        )
        pf_results = pf.run()
        pf_metrics = pf.report()

        if save_plots:
            pf.plot(
                save_path=str(project_path("results", f"portfolio_{strategy}.html"))
            )

        portfolio_results[strategy] = {
            "results": pf_results,
            "metrics": pf_metrics,
        }
        logger.info(
            f"  ✔  {strategy}: "
            f"Sharpe={pf_metrics['portfolio']['Sharpe Ratio']:.3f}  "
            f"Alpha={pf_metrics['alpha']:+.1f}%"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Final Summary
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t_start

    logger.info("\n" + "=" * 65)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Total time : {elapsed:.1f}s")
    logger.info(f"  Tickers    : {len(valid_tickers)}")
    logger.info(f"  Models     : {len(models_dict)} trained + saved")
    logger.info(f"  Charts     : results/*.html")
    logger.info("=" * 65)

    # Summary table
    summary_rows = []
    for ticker in valid_tickers:
        bm = backtest_dict[ticker]["metrics"]
        ml = metrics_dict[ticker]
        row = {"Ticker": ticker}
        if task == "regression":
            row["RMSE"]   = ml.get("RMSE", "—")
            row["DirAcc"] = f"{ml.get('DirAcc_%', 0):.1f}%"
        else:
            row["Accuracy"] = f"{ml.get('Accuracy', 0)*100:.1f}%"
            row["F1"]       = f"{ml.get('F1', 0):.3f}"
        row["Sharpe"] = bm["strategy"]["Sharpe Ratio"]
        row["Alpha%"] = f"{bm['alpha']:+.1f}"
        row["MDD%"]   = bm["strategy"]["Max Drawdown %"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
    print("\n" + "═"*65)
    print("  Per-Ticker Summary")
    print("═"*65)
    print(summary_df.to_string())
    print("═"*65)

    # Save summary CSV
    summary_path = project_path("results", "pipeline_summary.csv")
    summary_df.to_csv(summary_path)
    logger.success(f"Summary saved → {summary_path}")

    return {
        "models":            models_dict,
        "features":          features_dict,
        "backtest_results":  backtest_dict,
        "portfolio_results": portfolio_results,
        "summary":           summary_df,
    }


if __name__ == "__main__":
    run_full_pipeline()