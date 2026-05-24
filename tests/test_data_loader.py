"""
tests/test_data_loader.py
─────────────────────────
Unit tests for NSEDataLoader and helper functions.
Run with: pytest tests/test_data_loader.py -v

These tests use synthetic DataFrames — no network calls required.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Make sure src/ is importable when running from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import NSEDataLoader, temporal_split


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 500, start: str = "2020-01-01") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a DatetimeIndex."""
    dates = pd.bdate_range(start=start, periods=n)  # business days only
    close = 1000 + np.cumsum(np.random.randn(n) * 10)
    high  = close + np.abs(np.random.randn(n) * 5)
    low   = close - np.abs(np.random.randn(n) * 5)
    open_ = close + np.random.randn(n) * 3
    vol   = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": vol,
    }, index=dates)
    df.index.name = "Date"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Tests: temporal_split
# ══════════════════════════════════════════════════════════════════════════════

class TestTemporalSplit:

    def test_sizes_sum_to_total(self):
        df = make_ohlcv(1000)
        train, val, test = temporal_split(df, test_ratio=0.15, val_ratio=0.10)
        assert len(train) + len(val) + len(test) == len(df)

    def test_no_overlap(self):
        df = make_ohlcv(1000)
        train, val, test = temporal_split(df)
        assert train.index.max() < val.index.min(), "Train and Val overlap!"
        assert val.index.max() < test.index.min(), "Val and Test overlap!"

    def test_chronological_order(self):
        df = make_ohlcv(1000)
        train, val, test = temporal_split(df)
        assert train.index.is_monotonic_increasing
        assert val.index.is_monotonic_increasing
        assert test.index.is_monotonic_increasing

    def test_train_is_largest_partition(self):
        df = make_ohlcv(1000)
        train, val, test = temporal_split(df)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_approximate_ratios(self):
        df = make_ohlcv(1000)
        train, val, test = temporal_split(df, test_ratio=0.15, val_ratio=0.10)
        n = len(df)
        assert abs(len(test) / n - 0.15) < 0.02
        assert abs(len(val)  / n - 0.10) < 0.02


# ══════════════════════════════════════════════════════════════════════════════
# Tests: NSEDataLoader._validate (using synthetic data, no downloads)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:

    def setup_method(self):
        """Create a loader instance (no download will be triggered)."""
        self.loader = NSEDataLoader.__new__(NSEDataLoader)
        self.loader.ticker = "TEST.NS"

    def test_clean_df_passes(self):
        df = make_ohlcv(300)
        result = self.loader._validate(df)
        assert len(result) > 0

    def test_raises_on_missing_required_column(self):
        df = make_ohlcv(300).drop(columns=["Volume"])
        with pytest.raises(ValueError, match="Missing columns"):
            self.loader._validate(df)

    def test_raises_on_too_few_rows(self):
        df = make_ohlcv(50)  # less than 200 minimum
        with pytest.raises(ValueError, match="only 50 rows"):
            self.loader._validate(df)

    def test_fixes_high_less_than_close(self):
        df = make_ohlcv(300)
        # Inject bad data: set High below Close for 5 rows
        df.iloc[10:15, df.columns.get_loc("High")] = df.iloc[10:15]["Close"] - 20
        result = self.loader._validate(df)
        assert (result["High"] >= result["Close"]).all(), "High < Close not fixed"

    def test_fixes_low_greater_than_close(self):
        df = make_ohlcv(300)
        df.iloc[10:15, df.columns.get_loc("Low")] = df.iloc[10:15]["Close"] + 20
        result = self.loader._validate(df)
        assert (result["Low"] <= result["Close"]).all(), "Low > Close not fixed"

    def test_drops_all_nan_rows(self):
        df = make_ohlcv(300)
        df.iloc[5, :] = np.nan
        result = self.loader._validate(df)
        assert result.isnull().sum().sum() == 0

    def test_fills_zero_volume(self):
        df = make_ohlcv(300)
        df.iloc[10, df.columns.get_loc("Volume")] = 0
        result = self.loader._validate(df)
        assert (result["Volume"] > 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Tests: _normalise_columns
# ══════════════════════════════════════════════════════════════════════════════

class TestColumnNormalisation:

    def test_flattens_multiindex(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame(
            np.random.rand(5, 5),
            index=idx,
            columns=pd.MultiIndex.from_tuples([
                ("Open", ""), ("High", ""), ("Low", ""),
                ("Close", ""), ("Volume", ""),
            ]),
        )
        result = NSEDataLoader._normalise_columns(df)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_lowercase_columns_standardised(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = NSEDataLoader._normalise_columns(df)
        assert "Close" in result.columns
        assert "Volume" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# Tests: _clean_index
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanIndex:

    def test_removes_timezone(self):
        dates = pd.bdate_range("2020-01-01", periods=10, tz="Asia/Kolkata")
        df = pd.DataFrame({"Close": range(10)}, index=dates)
        result = NSEDataLoader._clean_index(df)
        assert result.index.tz is None

    def test_sorts_ascending(self):
        dates = pd.bdate_range("2020-01-01", periods=10)[::-1]  # reversed
        df = pd.DataFrame({"Close": range(10)}, index=dates)
        result = NSEDataLoader._clean_index(df)
        assert result.index.is_monotonic_increasing

    def test_index_name_is_date(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame({"Close": range(10)}, index=dates)
        result = NSEDataLoader._clean_index(df)
        assert result.index.name == "Date"