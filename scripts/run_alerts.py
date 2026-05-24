"""
scripts/run_alerts.py
──────────────────────
Run this every morning to get daily prediction alerts.

Schedule it with Windows Task Scheduler to run at 8:30 AM every day.

Usage:
    python scripts/run_alerts.py

Setup for WhatsApp:
    1. pip install twilio
    2. Add to config.yaml (see src/alerts.py for full instructions)

Setup for Email:
    1. Enable Gmail 2FA
    2. Generate App Password at myaccount.google.com
    3. Add to config.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib

from src.utils import load_config, setup_logger, project_path
from src.feature_engineering import FeatureEngineer
from src.regime_detection import RegimeDetector
from src.alerts import send_daily_alerts
from loguru import logger
import yfinance as yf


def get_predictions(cfg: dict, task: str = "regression") -> tuple[dict, dict]:
    """
    Load all trained models and generate predictions for all tickers.
    Returns (predictions_dict, multiday_dict)
    """
    tickers     = cfg["data"]["tickers"]
    predictions = {}
    multiday    = {}
    rd          = RegimeDetector()

    for ticker in tickers:
        safe = ticker.replace(".", "_")
        model_path = project_path("models", f"xgboost_{task}_{safe}.pkl")

        if not model_path.exists():
            logger.warning(f"No model for {ticker}, skipping")
            continue

        try:
            # Load model
            obj      = joblib.load(model_path)
            pipeline = obj.get("pipeline") if isinstance(obj, dict) else obj

            # Fetch latest data
            df_raw = yf.download(ticker, period="2y",
                                 auto_adjust=True, progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = [c[0] for c in df_raw.columns]
            df_raw.index = pd.to_datetime(df_raw.index)
            if df_raw.index.tz is not None:
                df_raw.index = df_raw.index.tz_localize(None)

            if len(df_raw) < 200:
                continue

            # Build features
            fe      = FeatureEngineer(df_raw, config=cfg)
            df_feat = fe.build()

            feature_cols = FeatureEngineer.get_feature_cols(df_feat)
            X_latest     = df_feat[feature_cols].iloc[[-1]]
            close_price  = float(df_feat["Close"].iloc[-1])

            # Tomorrow prediction
            pred      = float(pipeline.predict(X_latest)[0])
            direction = "UP" if pred > 0 else "DOWN"

            predictions[ticker] = {
                "direction":  direction,
                "score":      round(pred, 4),
                "price":      close_price,
                "confidence": abs(pred),
            }

            # Multi-day predictions
            multiday[ticker] = {}
            for horizon, col in [("3d", "Target_Dir_3d"), ("5d", "Target_Dir_5d")]:
                md_path = project_path("models", f"xgboost_classification_{safe}_{horizon}.pkl")
                if md_path.exists():
                    md_obj  = joblib.load(md_path)
                    md_pipe = md_obj.get("pipeline") if isinstance(md_obj, dict) else md_obj
                    md_pred = float(md_pipe.predict(X_latest)[0])
                    multiday[ticker][horizon] = "UP" if md_pred > 0.5 else "DOWN"
                else:
                    # Fallback: use sign of regression score as proxy
                    multiday[ticker][horizon] = direction

            logger.info(f"  {ticker}: {direction} ({pred:+.4f})")

        except Exception as e:
            logger.warning(f"  {ticker}: prediction failed — {e}")

    return predictions, multiday


def main():
    setup_logger(log_dir="logs")
    cfg = load_config()

    logger.info("=" * 55)
    logger.info("  NSE Alpha — Daily Alert Runner")
    logger.info("=" * 55)

    # Generate predictions
    logger.info("Generating predictions for all tickers...")
    predictions, multiday = get_predictions(cfg, task="regression")

    if not predictions:
        logger.error("No predictions generated. Train models first.")
        return

    logger.info(f"\n{len(predictions)} tickers predicted:")
    bullish = sum(1 for v in predictions.values() if v["direction"] == "UP")
    bearish = len(predictions) - bullish
    logger.info(f"  🟢 Bullish: {bullish}  🔴 Bearish: {bearish}")

    # Detect market regime using NIFTY 50
    logger.info("\nDetecting market regime...")
    regime_info = None
    try:
        nifty = yf.download("^NSEI", period="1y",
                            auto_adjust=True, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = [c[0] for c in nifty.columns]
        nifty.index = pd.to_datetime(nifty.index)
        if nifty.index.tz is not None:
            nifty.index = nifty.index.tz_localize(None)

        if len(nifty) >= 50:
            fe_nifty    = FeatureEngineer(nifty, config=cfg)
            df_nifty    = fe_nifty.build()
            rd          = RegimeDetector()
            regime_info = rd.detect(df_nifty)
            logger.info(
                f"Regime: {regime_info['emoji']} {regime_info['regime']} "
                f"({regime_info['confidence']:.0%})"
            )
    except Exception as e:
        logger.warning(f"Regime detection failed: {e}")

    # Send alerts
    logger.info("\nSending alerts...")
    results = send_daily_alerts(
        predictions=predictions,
        config=cfg,
        regime_info=regime_info,
        multiday=multiday,
    )

    logger.info("\n" + "=" * 55)
    logger.info(f"  Email:     {'✔ Sent' if results.get('email') else '✘ Not sent'}")
    logger.info(f"  WhatsApp:  {'✔ Sent' if results.get('whatsapp') else '✘ Not sent'}")
    logger.info("=" * 55)

    # Print preview to console
    logger.info("\nTop Bullish Picks:")
    top_bull = sorted(
        [(t, v) for t, v in predictions.items() if v["direction"] == "UP"],
        key=lambda x: x[1]["score"], reverse=True
    )[:5]
    for t, v in top_bull:
        print(f"  ▲ {t:<20} ₹{v['price']:>8,.0f}  score: {v['score']:+.4f}")

    logger.info("\nTop Bearish Picks:")
    top_bear = sorted(
        [(t, v) for t, v in predictions.items() if v["direction"] == "DOWN"],
        key=lambda x: x[1]["score"]
    )[:5]
    for t, v in top_bear:
        print(f"  ▼ {t:<20} ₹{v['price']:>8,.0f}  score: {v['score']:+.4f}")


if __name__ == "__main__":
    main()