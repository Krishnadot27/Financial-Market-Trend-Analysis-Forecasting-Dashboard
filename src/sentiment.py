"""
src/sentiment.py
─────────────────
FinBERT-based news sentiment analysis for NSE stocks.

FinBERT is a BERT model fine-tuned specifically on financial text.
It outputs three classes: POSITIVE, NEGATIVE, NEUTRAL

Why sentiment matters for NSE:
  - Earnings surprises, RBI policy, FII flows move Indian markets
  - News sentiment leads price by 1-2 days in emerging markets
  - Combining sentiment + technical features improves direction accuracy

Usage:
    from src.sentiment import SentimentAnalyser, fetch_news_sentiment

    sa = SentimentAnalyser()
    score = sa.analyse("Reliance beats Q3 profit estimates by 12%")
    # → {'label': 'positive', 'score': 0.94}

    df_sentiment = fetch_news_sentiment('RELIANCE', days=30)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import os

import numpy as np
import pandas as pd
from loguru import logger


class SentimentAnalyser:
    """
    FinBERT sentiment analysis for financial news headlines.

    Uses the ProsusAI/finbert model from HuggingFace.
    Falls back to a rule-based lexicon if transformers not installed.

    Parameters
    ----------
    model_name : HuggingFace model ID (default: ProsusAI/finbert)
    device     : 'cpu' | 'cuda' | 'mps'
    batch_size : Number of texts to process at once
    """

    # Financial lexicon for rule-based fallback
    POSITIVE_WORDS = {
        "beat", "beats", "surpass", "strong", "record", "profit",
        "growth", "upgrade", "buy", "outperform", "rally", "gain",
        "positive", "boost", "rise", "rises", "raised", "higher",
        "exceed", "exceeds", "upbeat", "optimistic", "robust",
        "breakout", "bullish", "recovery", "rebound",
    }
    NEGATIVE_WORDS = {
        "miss", "misses", "weak", "loss", "losses", "downgrade",
        "sell", "underperform", "fall", "falls", "decline", "cut",
        "negative", "concern", "lower", "drop", "drops", "risk",
        "warning", "bearish", "slowdown", "disappoint", "disappoints",
        "caution", "volatile", "uncertain",
    }

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device:     str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self.pipeline   = None
        self._use_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        """Load FinBERT pipeline, fall back to lexicon if unavailable."""
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading FinBERT: {self.model_name}")
            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                device=-1 if self.device == "cpu" else 0,
                truncation=True,
                max_length=512,
            )
            logger.success("FinBERT loaded successfully")
        except Exception as e:
            logger.warning(
                f"FinBERT not available ({e}). "
                "Using rule-based lexicon fallback. "
                "Install with: pip install transformers torch"
            )
            self._use_fallback = True

    def analyse(self, text: str) -> dict:
        """
        Analyse sentiment of a single text.

        Returns
        -------
        dict with keys: label ('positive'/'negative'/'neutral'), score (0-1)
        """
        if not text or len(text.strip()) < 3:
            return {"label": "neutral", "score": 0.5}

        if self._use_fallback:
            return self._lexicon_sentiment(text)

        try:
            result = self.pipeline(text[:512])[0]
            return {
                "label": result["label"].lower(),
                "score": round(float(result["score"]), 4),
            }
        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")
            return self._lexicon_sentiment(text)

    def analyse_batch(self, texts: list[str]) -> list[dict]:
        """
        Analyse a batch of texts efficiently.
        Processes in chunks of batch_size.
        """
        if not texts:
            return []

        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            if self._use_fallback:
                results.extend([self._lexicon_sentiment(t) for t in chunk])
            else:
                try:
                    batch_results = self.pipeline(
                        [t[:512] for t in chunk],
                        truncation=True,
                    )
                    results.extend([
                        {"label": r["label"].lower(), "score": round(float(r["score"]), 4)}
                        for r in batch_results
                    ])
                except Exception as e:
                    logger.warning(f"Batch inference failed: {e}")
                    results.extend([self._lexicon_sentiment(t) for t in chunk])

        return results

    def _lexicon_sentiment(self, text: str) -> dict:
        """Rule-based fallback using financial keyword lexicon."""
        words  = set(text.lower().split())
        pos    = len(words & self.POSITIVE_WORDS)
        neg    = len(words & self.NEGATIVE_WORDS)
        total  = pos + neg

        if total == 0:
            return {"label": "neutral", "score": 0.5}
        elif pos > neg:
            score = 0.5 + 0.4 * (pos / total)
            return {"label": "positive", "score": round(score, 4)}
        else:
            score = 0.5 + 0.4 * (neg / total)
            return {"label": "negative", "score": round(score, 4)}

    def sentiment_score(self, text: str) -> float:
        """
        Return a single float: +1.0 (very positive) to -1.0 (very negative).
        Useful as a feature for ML models.
        """
        result = self.analyse(text)
        score  = result["score"]
        if result["label"] == "positive":
            return round(score, 4)
        elif result["label"] == "negative":
            return round(-score, 4)
        return 0.0

    def analyse_headlines(
        self,
        headlines: list[str],
        dates:     Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Analyse a list of headlines and return a DataFrame.

        Columns: headline, label, score, sentiment_score, date (if provided)
        """
        results = self.analyse_batch(headlines)
        df = pd.DataFrame({
            "headline":       headlines,
            "label":          [r["label"] for r in results],
            "confidence":     [r["score"] for r in results],
            "sentiment_score": [
                r["score"] if r["label"] == "positive"
                else (-r["score"] if r["label"] == "negative" else 0.0)
                for r in results
            ],
        })
        if dates is not None:
            df["date"] = dates
            df = df.set_index("date").sort_index()
        return df


# ══════════════════════════════════════════════════════════════════════════════
# News fetcher
# ══════════════════════════════════════════════════════════════════════════════

def fetch_news_headlines(
    company_name: str,
    days:         int = 30,
    api_key:      Optional[str] = None,
) -> list[dict]:
    """
    Fetch recent news headlines for a company via NewsAPI.

    Returns list of dicts: {title, description, publishedAt, source}

    Parameters
    ----------
    company_name : e.g. 'Reliance Industries', 'TCS', 'Infosys'
    days         : How many days back to fetch
    api_key      : NewsAPI key (or set NEWS_API_KEY env var)
    """
    api_key = api_key or os.getenv("NEWS_API_KEY")
    if not api_key:
        logger.warning(
            "NEWS_API_KEY not set. Using sample headlines for demo. "
            "Get a free key at newsapi.org"
        )
        return _get_sample_headlines(company_name)

    try:
        import requests
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        f"{company_name} NSE India stock",
            "from":     from_date,
            "sortBy":   "publishedAt",
            "language": "en",
            "pageSize": 50,
            "apiKey":   api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        logger.info(f"Fetched {len(articles)} headlines for {company_name}")
        return articles
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed: {e}. Using sample headlines.")
        return _get_sample_headlines(company_name)


def _get_sample_headlines(company_name: str) -> list[dict]:
    """Sample financial headlines for demo/testing when no API key."""
    base = datetime.now()
    return [
        {"title": f"{company_name} Q3 profit beats estimates by 8%",
         "publishedAt": (base - timedelta(days=1)).isoformat()},
        {"title": f"Analysts upgrade {company_name} target price",
         "publishedAt": (base - timedelta(days=2)).isoformat()},
        {"title": f"{company_name} announces new digital transformation initiative",
         "publishedAt": (base - timedelta(days=3)).isoformat()},
        {"title": f"FII buying continues in {company_name} shares",
         "publishedAt": (base - timedelta(days=4)).isoformat()},
        {"title": f"{company_name} revenue growth slows amid global headwinds",
         "publishedAt": (base - timedelta(days=5)).isoformat()},
        {"title": f"RBI rate decision weighs on {company_name} outlook",
         "publishedAt": (base - timedelta(days=6)).isoformat()},
        {"title": f"{company_name} board approves share buyback programme",
         "publishedAt": (base - timedelta(days=7)).isoformat()},
        {"title": f"Weak demand outlook drags {company_name} shares lower",
         "publishedAt": (base - timedelta(days=8)).isoformat()},
    ]


def fetch_news_sentiment(
    company_name: str,
    ticker:       str,
    days:         int = 30,
    api_key:      Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch news + run FinBERT sentiment → return daily sentiment DataFrame.

    The daily sentiment score can be used as an additional feature
    or as a standalone trading signal.

    Returns
    -------
    pd.DataFrame with columns: date, headline, label, confidence,
                               sentiment_score, rolling_3d_sentiment
    """
    articles  = fetch_news_headlines(company_name, days, api_key)
    analyser  = SentimentAnalyser()

    headlines = [a.get("title", "") or "" for a in articles]
    dates     = [a.get("publishedAt", "")[:10] for a in articles]

    df = analyser.analyse_headlines(headlines, dates)

    # Rolling 3-day sentiment average
    df["rolling_3d"] = (
        df["sentiment_score"]
        .rolling(3, min_periods=1)
        .mean()
    )

    logger.info(
        f"Sentiment summary for {company_name}:\n"
        f"  Positive: {(df['label']=='positive').sum()}\n"
        f"  Negative: {(df['label']=='negative').sum()}\n"
        f"  Neutral:  {(df['label']=='neutral').sum()}\n"
        f"  Avg score: {df['sentiment_score'].mean():.3f}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Merge sentiment with price features
# ══════════════════════════════════════════════════════════════════════════════

def add_sentiment_features(
    df_features: pd.DataFrame,
    df_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join daily sentiment scores onto the feature DataFrame.

    Handles date alignment — sentiment published after market close
    affects next-day trading. Uses forward-fill for missing days.

    Parameters
    ----------
    df_features  : Feature matrix from FeatureEngineer.build()
    df_sentiment : Output of fetch_news_sentiment()

    Returns
    -------
    df_features with additional columns:
      - sentiment_score     (daily FinBERT score)
      - sentiment_rolling3d (3-day rolling average)
      - sentiment_positive  (binary: positive day)
      - sentiment_negative  (binary: negative day)
    """
    # Resample sentiment to daily (average if multiple headlines/day)
    sent_copy = df_sentiment.copy()
    sent_copy.index = pd.to_datetime(sent_copy.index)
    daily_sent = sent_copy["sentiment_score"].resample("D").mean()
    daily_roll = sent_copy["rolling_3d"].resample("D").mean()

    df_out = df_features.copy()

    # Align dates — shift by 1 day (yesterday's news → today's feature)
    df_out["sentiment_score"]    = daily_sent.shift(1).reindex(df_out.index, method="ffill")
    df_out["sentiment_rolling3d"] = daily_roll.shift(1).reindex(df_out.index, method="ffill")
    df_out["sentiment_positive"] = (df_out["sentiment_score"] > 0.2).astype(int)
    df_out["sentiment_negative"] = (df_out["sentiment_score"] < -0.2).astype(int)

    # Fill any remaining NaN with neutral (0)
    for col in ["sentiment_score", "sentiment_rolling3d",
                "sentiment_positive", "sentiment_negative"]:
        df_out[col] = df_out[col].fillna(0)

    logger.info(f"Added 4 sentiment features to feature matrix")
    return df_out