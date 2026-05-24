"""
tests/test_explainability_sentiment.py
────────────────────────────────────────
Tests for SHAP explainability and FinBERT sentiment modules.
Run: pytest tests/test_explainability_sentiment.py -v
"""

import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stubs
for mod_name in ["loguru", "yfinance", "tqdm"]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if mod_name == "loguru":
            class _L:
                def info(s,*a,**k): pass
                def debug(s,*a,**k): pass
                def warning(s,*a,**k): pass
                def error(s,*a,**k): pass
                def success(s,*a,**k): pass
            m.logger = _L()
        if mod_name == "tqdm":
            m.tqdm = lambda x, **k: x
        sys.modules[mod_name] = m

import numpy as np
import pandas as pd
import pytest

from src.sentiment import SentimentAnalyser, add_sentiment_features, _get_sample_headlines


# ══════════════════════════════════════════════════════════════════════════════
# SentimentAnalyser (lexicon fallback — no GPU/transformers required)
# ══════════════════════════════════════════════════════════════════════════════

class TestSentimentAnalyser:

    @pytest.fixture
    def sa(self):
        """Always uses lexicon fallback (no transformers needed)."""
        analyser = SentimentAnalyser.__new__(SentimentAnalyser)
        analyser._use_fallback = True
        analyser.batch_size    = 32
        analyser.pipeline      = None
        return analyser

    def test_positive_headline(self, sa):
        result = sa.analyse("Company beats profit estimates with record growth")
        assert result["label"] == "positive"
        assert result["score"] > 0.5

    def test_negative_headline(self, sa):
        result = sa.analyse("Stock falls on weak earnings and loss warning")
        assert result["label"] == "negative"
        assert result["score"] > 0.5

    def test_neutral_headline(self, sa):
        result = sa.analyse("Company announces annual general meeting date")
        assert result["label"] == "neutral"

    def test_empty_text(self, sa):
        result = sa.analyse("")
        assert result["label"] == "neutral"
        assert result["score"] == 0.5

    def test_score_range(self, sa):
        for text in [
            "Strong quarterly growth beats all estimates",
            "Weak results disappoint investors amid losses",
            "Board meeting scheduled for next quarter",
        ]:
            result = sa.analyse(text)
            assert 0.0 <= result["score"] <= 1.0

    def test_label_in_valid_set(self, sa):
        for text in ["profit up", "loss warning", "meeting today"]:
            result = sa.analyse(text)
            assert result["label"] in {"positive", "negative", "neutral"}

    def test_sentiment_score_positive_is_positive(self, sa):
        score = sa.sentiment_score("Record profit beats expectations strong growth")
        assert score > 0

    def test_sentiment_score_negative_is_negative(self, sa):
        score = sa.sentiment_score("Heavy losses disappoint weak results fall")
        assert score < 0

    def test_sentiment_score_range(self, sa):
        for text in ["profit rises record", "loss falls weak", "board meeting"]:
            score = sa.sentiment_score(text)
            assert -1.0 <= score <= 1.0

    def test_batch_same_as_individual(self, sa):
        texts = [
            "Record profits beat estimates strong growth",
            "Weak earnings disappoint loss warning",
            "Annual meeting scheduled",
        ]
        batch    = sa.analyse_batch(texts)
        individual = [sa.analyse(t) for t in texts]
        for b, i in zip(batch, individual):
            assert b["label"] == i["label"]

    def test_batch_returns_correct_length(self, sa):
        texts  = ["text " * i for i in range(1, 11)]
        results = sa.analyse_batch(texts)
        assert len(results) == len(texts)

    def test_analyse_headlines_returns_df(self, sa):
        headlines = ["Profit up", "Loss warning", "Normal day"]
        df = sa.analyse_headlines(headlines)
        assert isinstance(df, pd.DataFrame)
        assert "label" in df.columns
        assert "sentiment_score" in df.columns
        assert len(df) == 3

    def test_analyse_headlines_with_dates(self, sa):
        from datetime import datetime, timedelta
        base      = datetime.now()
        headlines = ["Good news", "Bad news", "Neutral news"]
        dates     = [(base - timedelta(days=i)).strftime("%Y-%m-%d")
                     for i in range(3)]
        df = sa.analyse_headlines(headlines, dates)
        assert df.index.name == "date" or isinstance(df.index[0], str)

    def test_sentiment_score_col_range(self, sa):
        headlines = [
            "Strong profit beats record growth rally",
            "Weak loss falls decline warning bearish",
            "Company holds annual meeting today",
        ]
        df = sa.analyse_headlines(headlines)
        assert (df["sentiment_score"] >= -1.0).all()
        assert (df["sentiment_score"] <= 1.0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Sample headlines
# ══════════════════════════════════════════════════════════════════════════════

class TestSampleHeadlines:

    def test_returns_list(self):
        result = _get_sample_headlines("Reliance")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_item_has_title(self):
        result = _get_sample_headlines("TCS")
        for item in result:
            assert "title" in item
            assert len(item["title"]) > 0

    def test_contains_company_name(self):
        result = _get_sample_headlines("Infosys")
        assert any("Infosys" in item["title"] for item in result)

    def test_has_published_date(self):
        result = _get_sample_headlines("HDFC")
        for item in result:
            assert "publishedAt" in item


# ══════════════════════════════════════════════════════════════════════════════
# add_sentiment_features
# ══════════════════════════════════════════════════════════════════════════════

class TestAddSentimentFeatures:

    @pytest.fixture
    def feature_df(self):
        """Minimal feature DataFrame with DatetimeIndex."""
        dates = pd.bdate_range("2023-01-01", periods=60)
        return pd.DataFrame(
            {"Close": 1000 + np.random.randn(60) * 10,
             "RSI_14": 50 + np.random.randn(60) * 10},
            index=dates,
        )

    @pytest.fixture
    def sentiment_df(self):
        """Synthetic sentiment DataFrame."""
        from datetime import datetime, timedelta
        base      = datetime(2023, 1, 1)
        dates     = [base + timedelta(days=i) for i in range(30)]
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        scores    = np.random.uniform(-1, 1, 30)
        rolling   = pd.Series(scores).rolling(3, min_periods=1).mean().values
        return pd.DataFrame({
            "sentiment_score": scores,
            "rolling_3d":      rolling,
            "label":           ["positive" if s > 0 else "negative" for s in scores],
        }, index=pd.DatetimeIndex(date_strs, name="date"))

    def test_adds_sentiment_columns(self, feature_df, sentiment_df):
        result = add_sentiment_features(feature_df, sentiment_df)
        for col in ["sentiment_score", "sentiment_rolling3d",
                    "sentiment_positive", "sentiment_negative"]:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nans_after_merge(self, feature_df, sentiment_df):
        result = add_sentiment_features(feature_df, sentiment_df)
        sent_cols = ["sentiment_score", "sentiment_rolling3d",
                     "sentiment_positive", "sentiment_negative"]
        assert result[sent_cols].isnull().sum().sum() == 0

    def test_binary_cols_are_binary(self, feature_df, sentiment_df):
        result = add_sentiment_features(feature_df, sentiment_df)
        for col in ["sentiment_positive", "sentiment_negative"]:
            assert set(result[col].unique()).issubset({0, 1})

    def test_original_cols_preserved(self, feature_df, sentiment_df):
        result = add_sentiment_features(feature_df, sentiment_df)
        assert "Close" in result.columns
        assert "RSI_14" in result.columns

    def test_row_count_unchanged(self, feature_df, sentiment_df):
        result = add_sentiment_features(feature_df, sentiment_df)
        assert len(result) == len(feature_df)