"""
scripts/run_explainability.py
──────────────────────────────
Step 6: SHAP explainability + news sentiment analysis.

Run from project root:
    python scripts/run_explainability.py

What it does:
  1. Loads trained XGBoost model
  2. Runs SHAP analysis — global feature importance + per-prediction explanation
  3. Fetches news headlines and runs FinBERT sentiment
  4. Shows how to add sentiment as ML features
  5. Saves all plots to results/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils import load_config, setup_logger, set_seed, project_path, ensure_dirs
from src.feature_engineering import FeatureEngineer
from src.model_training import XGBoostModel
from src.explainability import SHAPExplainer
from src.sentiment import SentimentAnalyser, fetch_news_sentiment, add_sentiment_features
from loguru import logger


# Ticker → company name mapping for news search
TICKER_TO_COMPANY = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS":      "Tata Consultancy Services",
    "INFY.NS":     "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS":"ICICI Bank",
    "WIPRO.NS":    "Wipro",
    "SBIN.NS":     "State Bank India",
}


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
    logger.info(f"  STEP 6 — Explainability + Sentiment: {ticker}")
    logger.info("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────────
    feat_path = project_path("data", "processed", f"{safe_ticker}_features.csv")
    if not feat_path.exists():
        logger.error("Run Steps 1-3 first")
        return

    df_feat = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
    X, y    = FeatureEngineer.get_X_y(df_feat, target=TARGET)

    # Temporal split
    n       = len(X)
    n_test  = int(n * cfg["split"]["test_ratio"])
    n_val   = int(n * cfg["split"]["val_ratio"])
    n_train = n - n_test - n_val
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test,  y_test  = X.iloc[n_train+n_val:], y.iloc[n_train+n_val:]

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = project_path("models", f"xgboost_{TASK}_{safe_ticker}.pkl")
    if not model_path.exists():
        logger.error("Train the model first: python scripts/run_model_training.py")
        return

    model = XGBoostModel.load(model_path)
    logger.info(f"Model loaded: {model_path.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: SHAP Explainability
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Part 1: SHAP Explainability ──")

    try:
        explainer = SHAPExplainer(model.pipeline, X_train)

        # Global feature importance
        logger.info("Generating SHAP summary plot...")
        importance = explainer.summary_plot(
            X_test,
            max_display=20,
            save_path=project_path("results", f"shap_summary_{safe_ticker}.png"),
        )

        # Bar chart
        explainer.bar_plot(
            X_test,
            save_path=project_path("results", f"shap_bar_{safe_ticker}.png"),
        )

        # Explain latest prediction
        logger.info("\nExplaining latest prediction...")
        explainer.explain_prediction(
            X_test.iloc[[-1]],
            ticker=ticker,
            save_path=project_path("results", f"shap_waterfall_{safe_ticker}.png"),
        )

        # Dependence plot for top feature
        top_feature = importance.index[0]
        logger.info(f"Dependence plot for top feature: {top_feature}")
        explainer.dependence_plot(
            X_test.iloc[:100],
            feature=top_feature,
            save_path=project_path("results", f"shap_dep_{safe_ticker}.png"),
        )

        logger.success("SHAP analysis complete — plots saved to results/")

    except ImportError:
        logger.warning(
            "SHAP not installed. Install with: pip install shap\n"
            "Skipping explainability section."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: FinBERT Sentiment
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Part 2: News Sentiment Analysis ──")

    company = TICKER_TO_COMPANY.get(ticker, ticker.replace(".NS", ""))
    logger.info(f"Fetching news for: {company}")

    df_sent = fetch_news_sentiment(
        company_name=company,
        ticker=ticker,
        days=30,
    )

    logger.info(f"\nSentiment results ({len(df_sent)} headlines):")
    print(df_sent[["headline", "label", "confidence", "sentiment_score"]].to_string())

    # Add sentiment features to the feature matrix
    logger.info("\n── Adding sentiment features to feature matrix ──")
    df_with_sent = add_sentiment_features(df_feat, df_sent)

    new_cols = ["sentiment_score", "sentiment_rolling3d",
                "sentiment_positive", "sentiment_negative"]
    logger.info(f"New columns added: {new_cols}")
    logger.info(f"\nLatest sentiment values:")
    for col in new_cols:
        val = df_with_sent[col].iloc[-1]
        print(f"  {col:<28} {val:.4f}")

    # Show sentiment distribution
    print(f"\nSentiment distribution:")
    print(df_sent["label"].value_counts().to_string())
    avg = df_sent["sentiment_score"].mean()
    print(f"\nOverall sentiment for {company}: "
          f"{'BULLISH 📈' if avg > 0.1 else 'BEARISH 📉' if avg < -0.1 else 'NEUTRAL ⚖'} "
          f"(score: {avg:+.3f})")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: Quick test — does sentiment improve model?
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Part 3: Sentiment Feature Impact Test ──")

    from src.model_training import XGBoostModel as XGB
    from src.evaluation import ModelEvaluator

    # Without sentiment
    X_base, y_base = FeatureEngineer.get_X_y(df_feat, target=TARGET)
    n2 = len(X_base)
    nt = int(n2 * cfg["split"]["test_ratio"])
    nv = int(n2 * cfg["split"]["val_ratio"])
    split = n2 - nt - nv

    X_tr1 = X_base.iloc[:split]; y_tr1 = y_base.iloc[:split]
    X_te1 = X_base.iloc[split+nv:]; y_te1 = y_base.iloc[split+nv:]

    # With sentiment
    X_sent, y_sent = FeatureEngineer.get_X_y(df_with_sent, target=TARGET)
    X_tr2 = X_sent.iloc[:split]; y_tr2 = y_sent.iloc[:split]
    X_te2 = X_sent.iloc[split+nv:]; y_te2 = y_sent.iloc[split+nv:]

    ev = ModelEvaluator(task=TASK)

    for label, Xtr, ytr, Xte, yte in [
        ("Without sentiment", X_tr1, y_tr1, X_te1, y_te1),
        ("With sentiment",    X_tr2, y_tr2, X_te2, y_te2),
    ]:
        m = XGB(task=TASK, n_estimators=200, learning_rate=0.05)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        if TASK == "regression":
            metrics = ev.regression_report(yte, pred, label=label)
        else:
            metrics = ev.classification_report(yte, pred, label=label)

    logger.info("\n" + "=" * 60)
    logger.info("  Step 6 complete!")
    logger.info("  ✔  SHAP plots saved → results/")
    logger.info("  ✔  Sentiment features added to pipeline")
    logger.info("  ✔  Model comparison with/without sentiment done")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()