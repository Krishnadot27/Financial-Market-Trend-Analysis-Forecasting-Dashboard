"""
tests/test_feature_engineering.py
───────────────────────────────────
Unit tests for FeatureEngineer.

Tests verify:
  - Each feature group produces expected columns
  - No lookahead bias (features use only past data)
  - No NaN leakage after build()
  - Target variables are correctly aligned
  - X/y split works correctly

Run with: pytest tests/test_feature_engineering.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

# ── Stub loguru so tests run without it installed ─────────────────────────────
import types, sys as _sys
if "loguru" not in _sys.modules:
    _loguru = types.ModuleType("loguru")
    class _L:
        def info(self,*a,**k): pass
        def debug(self,*a,**k): pass
        def warning(self,*a,**k): pass
        def error(self,*a,**k): pass
        def success(self,*a,**k): pass
    _loguru.logger = _L()
    _sys.modules["loguru"] = _loguru

from src.feature_engineering import FeatureEngineer


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """600 rows of realistic synthetic NSE-like OHLCV data."""
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 1000 + np.cumsum(np.random.randn(n) * 15)
    close = np.maximum(close, 100)   # keep prices positive
    high  = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low   = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    vol   = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def fe(sample_df):
    return FeatureEngineer(sample_df)


@pytest.fixture
def built_df(sample_df):
    fe = FeatureEngineer(sample_df)
    return fe.build()


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError, match="empty"):
            FeatureEngineer(pd.DataFrame())

    def test_raises_on_missing_column(self, sample_df):
        with pytest.raises(ValueError, match="missing"):
            FeatureEngineer(sample_df.drop(columns=["Volume"]))

    def test_log_return_computed_on_init(self, fe, sample_df):
        assert "LogReturn" in fe.df.columns
        assert len(fe.df) == len(sample_df)


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: Trend
# ══════════════════════════════════════════════════════════════════════════════

class TestTrend:

    def test_sma_columns_created(self, fe):
        fe.add_moving_averages()
        for w in [5, 10, 20, 50, 200]:
            assert f"SMA_{w}" in fe.df.columns, f"SMA_{w} missing"

    def test_ema_columns_created(self, fe):
        fe.add_moving_averages()
        for w in [5, 10, 20, 50]:
            assert f"EMA_{w}" in fe.df.columns

    def test_price_sma_ratio_near_zero_on_average(self, fe):
        fe.add_moving_averages()
        # Price/SMA ratio should be close to 0 on average (normalised)
        ratio = fe.df["Price_SMA20_ratio"].dropna()
        assert ratio.abs().mean() < 0.1

    def test_macd_hist_columns(self, fe):
        fe.add_macd()
        for col in ["MACD", "MACD_Signal", "MACD_Hist", "MACD_Cross"]:
            assert col in fe.df.columns

    def test_golden_cross_binary(self, fe):
        fe.add_moving_averages()
        if "GoldenCross" in fe.df.columns:
            vals = fe.df["GoldenCross"].dropna().unique()
            assert set(vals).issubset({0, 1})


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: Momentum
# ══════════════════════════════════════════════════════════════════════════════

class TestMomentum:

    def test_rsi_range(self, fe):
        fe.add_rsi()
        for p in [7, 14, 21]:
            rsi = fe.df[f"RSI_{p}"].dropna()
            assert (rsi >= 0).all() and (rsi <= 100).all(), f"RSI_{p} out of [0,100]"

    def test_rsi_overbought_oversold_binary(self, fe):
        fe.add_rsi()
        for p in [7, 14, 21]:
            vals = fe.df[f"RSI_{p}_overbought"].dropna().unique()
            assert set(vals).issubset({0, 1})

    def test_stochastic_range(self, fe):
        fe.add_stochastic()
        k = fe.df["Stoch_K"].dropna()
        assert (k >= 0).all() and (k <= 100).all()

    def test_roc_columns_created(self, fe):
        fe.add_roc()
        for p in [5, 10, 20]:
            assert f"ROC_{p}" in fe.df.columns

    def test_williams_r_range(self, fe):
        fe.add_williams_r()
        wr = fe.df["Williams_R"].dropna()
        assert (wr >= -100).all() and (wr <= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: Volatility
# ══════════════════════════════════════════════════════════════════════════════

class TestVolatility:

    def test_bb_columns_created(self, fe):
        fe.add_bollinger_bands()
        for col in ["BB_Upper", "BB_Lower", "BB_Mid", "BB_Width", "BB_PctB"]:
            assert col in fe.df.columns

    def test_bb_upper_above_lower(self, fe):
        fe.add_bollinger_bands()
        valid = fe.df[["BB_Upper", "BB_Lower"]].dropna()
        assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()

    def test_atr_positive(self, fe):
        fe.add_atr()
        atr = fe.df["ATR"].dropna()
        assert (atr > 0).all()

    def test_rolling_vol_positive(self, fe):
        fe.add_rolling_volatility()
        for w in [5, 10, 20]:
            vol = fe.df[f"Vol_{w}d"].dropna()
            assert (vol >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Group 4: Volume
# ══════════════════════════════════════════════════════════════════════════════

class TestVolume:

    def test_volume_columns_created(self, fe):
        fe.add_volume_features()
        for col in ["OBV", "VWAP_20", "Volume_ratio", "MFI", "Price_VWAP_ratio"]:
            assert col in fe.df.columns

    def test_mfi_range(self, fe):
        fe.add_volume_features()
        mfi = fe.df["MFI"].dropna()
        assert (mfi >= 0).all() and (mfi <= 100).all()

    def test_volume_surge_binary(self, fe):
        fe.add_volume_features()
        vals = fe.df["Volume_surge"].dropna().unique()
        assert set(vals).issubset({0, 1})


# ══════════════════════════════════════════════════════════════════════════════
# Group 5: Lag Features
# ══════════════════════════════════════════════════════════════════════════════

class TestLagFeatures:

    def test_lag_columns_created(self, fe):
        fe.add_lag_features()
        for lag in [1, 2, 3, 5, 10]:
            assert f"Close_lag{lag}" in fe.df.columns

    def test_lag1_is_previous_value(self, sample_df):
        fe = FeatureEngineer(sample_df)
        fe.add_lag_features()
        # Close_lag1[i] should equal Close[i-1]
        original = sample_df["Close"].values
        lag1     = fe.df["Close_lag1"].values
        # Compare from index 1 onward (first value is NaN)
        for i in range(1, 20):
            assert abs(lag1[i] - original[i-1]) < 1e-6, f"Lag1 mismatch at index {i}"

    def test_no_negative_lags(self, fe):
        """Ensure we never shift forward (negative lag = lookahead)."""
        fe.add_lag_features()
        lag_cols = [c for c in fe.df.columns if "_lag" in c]
        for col in lag_cols:
            # Extract lag number from column name
            lag_num = int(col.split("_lag")[-1])
            assert lag_num > 0, f"Non-positive lag found: {col}"


# ══════════════════════════════════════════════════════════════════════════════
# Group 7: Calendar
# ══════════════════════════════════════════════════════════════════════════════

class TestCalendar:

    def test_calendar_columns_created(self, fe):
        fe.add_calendar_features()
        for col in ["DayOfWeek", "Month", "Quarter", "IsMonthEnd"]:
            assert col in fe.df.columns

    def test_day_of_week_range(self, fe):
        fe.add_calendar_features()
        dow = fe.df["DayOfWeek"].unique()
        assert set(dow).issubset(set(range(7)))

    def test_month_range(self, fe):
        fe.add_calendar_features()
        assert fe.df["Month"].between(1, 12).all()

    def test_binary_flags_are_binary(self, fe):
        fe.add_calendar_features()
        for col in ["IsMonthEnd", "IsMonthStart", "IsMonday", "IsFriday"]:
            vals = fe.df[col].unique()
            assert set(vals).issubset({0, 1}), f"{col} is not binary"


# ══════════════════════════════════════════════════════════════════════════════
# Target Variables
# ══════════════════════════════════════════════════════════════════════════════

class TestTargets:

    def test_target_columns_created(self, fe):
        fe.add_targets()
        for col in ["Target_Price", "Target_Return", "Target_Dir"]:
            assert col in fe.df.columns

    def test_target_dir_is_binary(self, fe):
        fe.add_targets()
        vals = fe.df["Target_Dir"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_target_price_is_next_close(self, sample_df):
        """Target_Price at row i must equal Close at row i+1."""
        fe = FeatureEngineer(sample_df)
        fe.add_targets()
        closes  = fe.df["Close"].values
        targets = fe.df["Target_Price"].values
        for i in range(len(closes) - 2):
            if not np.isnan(targets[i]):
                assert abs(targets[i] - closes[i+1]) < 1e-6, \
                    f"Target_Price[{i}]={targets[i]} ≠ Close[{i+1}]={closes[i+1]}"

    def test_target_not_in_feature_cols(self, built_df):
        """Targets must NEVER appear in the feature set passed to the model."""
        feature_cols = FeatureEngineer.get_feature_cols(built_df)
        for t in ["Target_Price", "Target_Return", "Target_Dir"]:
            assert t not in feature_cols, f"TARGET LEAKAGE: {t} is in feature_cols!"


# ══════════════════════════════════════════════════════════════════════════════
# Full Build
# ══════════════════════════════════════════════════════════════════════════════

class TestBuild:

    def test_build_returns_dataframe(self, built_df):
        assert isinstance(built_df, pd.DataFrame)

    def test_build_no_nans_in_features(self, built_df):
        feature_cols = FeatureEngineer.get_feature_cols(built_df)
        null_counts  = built_df[feature_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0, \
            f"Features with NaNs after build():\n{cols_with_nulls}"

    def test_build_minimum_features(self, built_df):
        feature_cols = FeatureEngineer.get_feature_cols(built_df)
        assert len(feature_cols) >= 50, \
            f"Expected ≥50 features, got {len(feature_cols)}"

    def test_build_sufficient_rows(self, built_df):
        assert len(built_df) >= 300, \
            f"Too few rows after build(): {len(built_df)}"

    def test_index_still_sorted(self, built_df):
        assert built_df.index.is_monotonic_increasing

    def test_get_X_y_shapes_match(self, built_df):
        X, y = FeatureEngineer.get_X_y(built_df, target="Target_Dir")
        assert len(X) == len(y)
        assert X.shape[0] == len(built_df)

    def test_get_X_y_no_target_in_X(self, built_df):
        X, _ = FeatureEngineer.get_X_y(built_df, target="Target_Dir")
        for t in ["Target_Price", "Target_Return", "Target_Dir"]:
            assert t not in X.columns

    def test_get_X_y_raises_on_bad_target(self, built_df):
        with pytest.raises(ValueError, match="not found"):
            FeatureEngineer.get_X_y(built_df, target="NonExistentTarget")

    def test_no_lookahead_in_features(self, sample_df):
        """
        Critical anti-lookahead test:
        Corrupt the last 10 rows of Close and verify features
        at row -11 are NOT affected (they must be based on past only).
        """
        df_orig    = sample_df.copy()
        df_corrupt = sample_df.copy()
        df_corrupt.iloc[-10:, df_corrupt.columns.get_loc("Close")] *= 10  # huge shock

        fe_orig    = FeatureEngineer(df_orig).build()
        fe_corrupt = FeatureEngineer(df_corrupt).build()

        check_idx = -15   # row well before the corruption
        for col in ["SMA_20", "RSI_14", "MACD"]:
            if col in fe_orig.columns and col in fe_corrupt.columns:
                orig_val    = fe_orig.iloc[check_idx][col]
                corrupt_val = fe_corrupt.iloc[check_idx][col]
                assert abs(orig_val - corrupt_val) < 1e-6, \
                    f"LOOKAHEAD DETECTED in {col}! " \
                    f"orig={orig_val:.4f}, corrupt={corrupt_val:.4f}"