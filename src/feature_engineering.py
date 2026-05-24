"""
src/feature_engineering.py
───────────────────────────
Production-grade feature engineering for NSE stocks.

ALL features use only past data — zero lookahead bias.
Features are organised into 7 groups:
  1. Trend          — SMA, EMA, MACD, crossovers
  2. Momentum       — RSI, Stochastic, ROC, Williams %R
  3. Volatility     — Bollinger Bands, ATR, Keltner, historical vol
  4. Volume         — OBV, VWAP, Volume ratio, MFI
  5. Lag features   — Autoregressive lags of price, return, volume
  6. Rolling stats  — Rolling mean, std, skew, min/max of returns
  7. Calendar       — Day-of-week, month, quarter effects

Usage:
    from src.feature_engineering import FeatureEngineer

    fe  = FeatureEngineer(df, config=cfg)
    df_feat = fe.build()           # Run all steps, drop NaN rows
    X, y    = fe.get_X_y(df_feat, target='Target_Dir')
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import load_config, timer


# ══════════════════════════════════════════════════════════════════════════════
# FeatureEngineer
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Fluent-interface feature engineering pipeline.

    Parameters
    ----------
    df     : pd.DataFrame  Clean OHLCV DataFrame with DatetimeIndex
    config : dict          Loaded config.yaml (optional — uses defaults if None)

    Example
    -------
    fe      = FeatureEngineer(df, config=cfg)
    df_feat = fe.build()
    X, y    = fe.get_X_y(df_feat, target='Target_Dir')
    """

    # Columns that are raw inputs, never used as ML features
    RAW_COLS    = {"Open", "High", "Low", "Close", "Volume"}
    TARGET_COLS = {"Target_Price", "Target_Return", "Target_Dir"}

    def __init__(self, df: pd.DataFrame, config: Optional[dict] = None) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.df  = df.copy()
        self.cfg = config or {}
        self._feat_cfg = self.cfg.get("features", {})

        # Precompute log returns (used by many feature groups)
        self.df["LogReturn"] = np.log(
            self.df["Close"] / self.df["Close"].shift(1)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 1 — Trend Indicators
    # ══════════════════════════════════════════════════════════════════════════

    def add_moving_averages(self) -> "FeatureEngineer":
        """SMA and EMA for multiple windows + crossover signals."""
        sma_windows = self._feat_cfg.get("sma_windows", [5, 10, 20, 50, 200])
        ema_windows = self._feat_cfg.get("ema_windows", [5, 10, 20, 50])

        close = self.df["Close"]

        for w in sma_windows:
            self.df[f"SMA_{w}"] = close.rolling(w).mean()
            # Price relative to SMA — normalised, scale-free
            self.df[f"Price_SMA{w}_ratio"] = close / self.df[f"SMA_{w}"] - 1

        for w in ema_windows:
            self.df[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()
            self.df[f"Price_EMA{w}_ratio"] = close / self.df[f"EMA_{w}"] - 1

        # Golden Cross / Death Cross signals
        if "SMA_50" in self.df and "SMA_200" in self.df:
            self.df["GoldenCross"]    = (self.df["SMA_50"] > self.df["SMA_200"]).astype(int)
            self.df["SMA50_SMA200_ratio"] = self.df["SMA_50"] / self.df["SMA_200"] - 1

        # Short vs long EMA spread
        if "EMA_5" in self.df and "EMA_20" in self.df:
            self.df["EMA5_EMA20_spread"] = self.df["EMA_5"] / self.df["EMA_20"] - 1

        logger.debug("✔ Moving averages added")
        return self

    def add_macd(self) -> "FeatureEngineer":
        """
        MACD — Moving Average Convergence Divergence.
        Captures trend momentum and potential reversals.
        """
        fast   = self._feat_cfg.get("macd_fast",   12)
        slow   = self._feat_cfg.get("macd_slow",   26)
        signal = self._feat_cfg.get("macd_signal",  9)

        close = self.df["Close"]
        ema_f = close.ewm(span=fast,   adjust=False).mean()
        ema_s = close.ewm(span=slow,   adjust=False).mean()

        self.df["MACD"]        = ema_f - ema_s
        self.df["MACD_Signal"] = self.df["MACD"].ewm(span=signal, adjust=False).mean()
        self.df["MACD_Hist"]   = self.df["MACD"] - self.df["MACD_Signal"]

        # Normalised by price so it's comparable across stocks
        self.df["MACD_norm"]   = self.df["MACD"] / close

        # Bullish / Bearish signal: +1 when MACD > Signal, -1 otherwise
        self.df["MACD_Cross"]  = np.where(
            self.df["MACD"] > self.df["MACD_Signal"], 1, -1
        )
        # Histogram sign change (early reversal signal)
        self.df["MACD_Hist_sign"] = np.sign(self.df["MACD_Hist"])

        logger.debug("✔ MACD added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2 — Momentum Indicators
    # ══════════════════════════════════════════════════════════════════════════

    def add_rsi(self) -> "FeatureEngineer":
        """
        RSI — Relative Strength Index.
        Overbought (>70) / Oversold (<30) momentum oscillator.
        Uses Wilder's smoothing (EWM with com=period-1).
        """
        periods = self._feat_cfg.get("rsi_periods", [7, 14, 21])
        delta   = self.df["Close"].diff()
        gain    = delta.clip(lower=0)
        loss    = (-delta).clip(lower=0)

        for p in periods:
            avg_gain = gain.ewm(com=p - 1, min_periods=p).mean()
            avg_loss = loss.ewm(com=p - 1, min_periods=p).mean()
            rs       = avg_gain / avg_loss.replace(0, np.nan)
            rsi      = 100 - (100 / (1 + rs))
            self.df[f"RSI_{p}"] = rsi

            # Zone flags — useful as categorical signals
            self.df[f"RSI_{p}_overbought"] = (rsi > 70).astype(int)
            self.df[f"RSI_{p}_oversold"]   = (rsi < 30).astype(int)

        logger.debug("✔ RSI added")
        return self

    def add_stochastic(self) -> "FeatureEngineer":
        """
        Stochastic Oscillator %K and %D.
        Measures close position within the recent high-low range.
        """
        k_period = 14
        d_period = 3
        low_min  = self.df["Low"].rolling(k_period).min()
        high_max = self.df["High"].rolling(k_period).max()
        denom    = (high_max - low_min).replace(0, np.nan)

        self.df["Stoch_K"] = 100 * (self.df["Close"] - low_min) / denom
        self.df["Stoch_D"] = self.df["Stoch_K"].rolling(d_period).mean()
        self.df["Stoch_KD_diff"] = self.df["Stoch_K"] - self.df["Stoch_D"]

        logger.debug("✔ Stochastic added")
        return self

    def add_roc(self) -> "FeatureEngineer":
        """
        Rate of Change — percentage price change over N periods.
        Pure momentum signal.
        """
        for p in [5, 10, 20]:
            self.df[f"ROC_{p}"] = self.df["Close"].pct_change(p)

        logger.debug("✔ ROC added")
        return self

    def add_williams_r(self, period: int = 14) -> "FeatureEngineer":
        """
        Williams %R — momentum oscillator, inverse of Stochastic %K.
        Range: -100 (oversold) to 0 (overbought).
        """
        high_max = self.df["High"].rolling(period).max()
        low_min  = self.df["Low"].rolling(period).min()
        denom    = (high_max - low_min).replace(0, np.nan)
        self.df["Williams_R"] = -100 * (high_max - self.df["Close"]) / denom

        logger.debug("✔ Williams %R added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3 — Volatility Indicators
    # ══════════════════════════════════════════════════════════════════════════

    def add_bollinger_bands(self) -> "FeatureEngineer":
        """
        Bollinger Bands — dynamic support/resistance using rolling std.
        BB_PctB tells you WHERE price is within the bands (0=lower, 1=upper).
        BB_Width is a volatility proxy — widens in volatile markets.
        """
        window  = self._feat_cfg.get("bb_window", 20)
        num_std = self._feat_cfg.get("bb_std", 2)

        sma  = self.df["Close"].rolling(window).mean()
        std  = self.df["Close"].rolling(window).std()

        self.df["BB_Upper"] = sma + num_std * std
        self.df["BB_Lower"] = sma - num_std * std
        self.df["BB_Mid"]   = sma
        self.df["BB_Width"] = (self.df["BB_Upper"] - self.df["BB_Lower"]) / sma
        self.df["BB_PctB"]  = (
            (self.df["Close"] - self.df["BB_Lower"]) /
            (self.df["BB_Upper"] - self.df["BB_Lower"]).replace(0, np.nan)
        )
        # Squeeze: BB width at multi-month low → potential breakout ahead
        self.df["BB_Squeeze"] = (
            self.df["BB_Width"] < self.df["BB_Width"].rolling(125).quantile(0.2)
        ).astype(int)

        logger.debug("✔ Bollinger Bands added")
        return self

    def add_atr(self) -> "FeatureEngineer":
        """
        ATR — Average True Range.
        Measures market volatility independent of price direction.
        ATR_pct normalises by price → comparable across stocks.
        """
        window = self._feat_cfg.get("atr_window", 14)
        high, low, prev_close = (
            self.df["High"],
            self.df["Low"],
            self.df["Close"].shift(1),
        )
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        self.df["ATR"]     = tr.rolling(window).mean()
        self.df["ATR_pct"] = self.df["ATR"] / self.df["Close"]

        # ATR trend: is volatility expanding or contracting?
        self.df["ATR_expanding"] = (
            self.df["ATR"] > self.df["ATR"].shift(5)
        ).astype(int)

        logger.debug("✔ ATR added")
        return self

    def add_rolling_volatility(self) -> "FeatureEngineer":
        """
        Historical volatility: rolling std of log returns, annualised.
        Multiple windows capture different volatility regimes.
        """
        windows = self._feat_cfg.get("vol_windows", [5, 10, 20])
        for w in windows:
            self.df[f"Vol_{w}d"] = (
                self.df["LogReturn"].rolling(w).std() * np.sqrt(252)
            )
        # Volatility ratio: short-term vs long-term (vol regime indicator)
        if "Vol_5d" in self.df and "Vol_20d" in self.df:
            self.df["Vol_ratio_5_20"] = self.df["Vol_5d"] / self.df["Vol_20d"].replace(0, np.nan)

        logger.debug("✔ Rolling volatility added")
        return self

    def add_keltner_channels(self, ema_period: int = 20, atr_mult: float = 2.0) -> "FeatureEngineer":
        """
        Keltner Channels — volatility envelope based on EMA + ATR.
        Complementary to Bollinger Bands.
        """
        if "ATR" not in self.df.columns:
            self.add_atr()

        ema = self.df["Close"].ewm(span=ema_period, adjust=False).mean()
        self.df["KC_Upper"] = ema + atr_mult * self.df["ATR"]
        self.df["KC_Lower"] = ema - atr_mult * self.df["ATR"]
        self.df["KC_PctPos"] = (
            (self.df["Close"] - self.df["KC_Lower"]) /
            (self.df["KC_Upper"] - self.df["KC_Lower"]).replace(0, np.nan)
        )

        logger.debug("✔ Keltner Channels added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4 — Volume Indicators
    # ══════════════════════════════════════════════════════════════════════════

    def add_volume_features(self) -> "FeatureEngineer":
        """
        Volume-based signals: OBV, VWAP, Volume ratio, MFI.
        Volume confirms price moves — high volume breakouts are more reliable.
        """
        close, volume = self.df["Close"], self.df["Volume"]

        # Volume moving averages & ratio
        self.df["Volume_SMA20"]  = volume.rolling(20).mean()
        self.df["Volume_ratio"]  = volume / self.df["Volume_SMA20"].replace(0, np.nan)
        self.df["Volume_surge"]  = (self.df["Volume_ratio"] > 2.0).astype(int)

        # OBV — On-Balance Volume: cumulative volume signed by price direction
        self.df["OBV"] = (np.sign(close.diff()) * volume).cumsum()
        self.df["OBV_SMA20"] = self.df["OBV"].rolling(20).mean()
        self.df["OBV_trend"]  = (self.df["OBV"] > self.df["OBV_SMA20"]).astype(int)

        # VWAP (rolling 20-day) — institutional benchmark price
        self.df["VWAP_20"] = (
            (close * volume).rolling(20).sum() /
            volume.rolling(20).sum().replace(0, np.nan)
        )
        self.df["Price_VWAP_ratio"] = close / self.df["VWAP_20"] - 1

        # Money Flow Index (MFI) — volume-weighted RSI
        typical_price = (self.df["High"] + self.df["Low"] + close) / 3
        money_flow    = typical_price * volume
        tp_delta      = typical_price.diff()

        pos_flow = money_flow.where(tp_delta > 0, 0).rolling(14).sum()
        neg_flow = money_flow.where(tp_delta < 0, 0).rolling(14).sum()
        mfi_ratio = pos_flow / neg_flow.replace(0, np.nan)
        self.df["MFI"] = 100 - (100 / (1 + mfi_ratio))

        logger.debug("✔ Volume features added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 5 — Lag Features (Autoregressive)
    # ══════════════════════════════════════════════════════════════════════════

    def add_lag_features(self) -> "FeatureEngineer":
        """
        Lag features for autoregressive modelling.
        Creates past values of key signals as features.
        IMPORTANT: shift(n) = n-period LAG → no lookahead.
        """
        lag_cols    = self._feat_cfg.get("lag_cols",    ["Close", "Volume", "LogReturn"])
        lag_periods = self._feat_cfg.get("lag_periods", [1, 2, 3, 5, 10])

        for col in lag_cols:
            if col not in self.df.columns:
                continue
            for lag in lag_periods:
                self.df[f"{col}_lag{lag}"] = self.df[col].shift(lag)

        # Lagged RSI and MACD (top predictors in practice)
        for col in ["RSI_14", "MACD_Hist", "BB_PctB"]:
            if col in self.df.columns:
                for lag in [1, 2, 3]:
                    self.df[f"{col}_lag{lag}"] = self.df[col].shift(lag)

        logger.debug("✔ Lag features added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 6 — Rolling Statistics
    # ══════════════════════════════════════════════════════════════════════════

    def add_rolling_stats(self) -> "FeatureEngineer":
        """
        Rolling statistical moments of log returns.
        Captures distribution shape changes — regime detection signals.
        """
        windows = self._feat_cfg.get("rolling_windows", [5, 10, 20])

        for w in windows:
            ret = self.df["LogReturn"]
            self.df[f"Return_mean_{w}"]  = ret.rolling(w).mean()
            self.df[f"Return_std_{w}"]   = ret.rolling(w).std()
            self.df[f"Return_skew_{w}"]  = ret.rolling(w).skew()
            self.df[f"Return_kurt_{w}"]  = ret.rolling(w).kurt()

            # Price range features
            self.df[f"Price_max_{w}"]    = self.df["Close"].rolling(w).max()
            self.df[f"Price_min_{w}"]    = self.df["Close"].rolling(w).min()
            self.df[f"Price_range_{w}"]  = (
                self.df[f"Price_max_{w}"] - self.df[f"Price_min_{w}"]
            ) / self.df["Close"]

            # Close position within recent range [0=bottom, 1=top]
            self.df[f"Close_rank_{w}"] = (
                (self.df["Close"] - self.df[f"Price_min_{w}"]) /
                (self.df[f"Price_max_{w}"] - self.df[f"Price_min_{w}"]).replace(0, np.nan)
            )

        logger.debug("✔ Rolling stats added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 7 — Calendar / Seasonal Features
    # ══════════════════════════════════════════════════════════════════════════

    def add_calendar_features(self) -> "FeatureEngineer":
        """
        Calendar effects in Indian markets:
        - Monday effect (often weak open after weekend)
        - Month-end rebalancing (institutional buying)
        - Budget session months (Feb/Mar = high volatility in NSE)
        - Quarter-end effects
        """
        idx = self.df.index

        self.df["DayOfWeek"]    = idx.dayofweek          # 0=Mon, 4=Fri
        self.df["Month"]        = idx.month
        self.df["Quarter"]      = idx.quarter
        self.df["IsMonthEnd"]   = idx.is_month_end.astype(int)
        self.df["IsMonthStart"] = idx.is_month_start.astype(int)
        self.df["IsQtrEnd"]     = idx.is_quarter_end.astype(int)

        # India-specific: Budget month (Feb) and expiry week (last Thu of month)
        self.df["IsBudgetMonth"] = (idx.month == 2).astype(int)
        self.df["IsMonday"]      = (idx.dayofweek == 0).astype(int)
        self.df["IsFriday"]      = (idx.dayofweek == 4).astype(int)

        logger.debug("✔ Calendar features added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # Target Variable Creation
    # ══════════════════════════════════════════════════════════════════════════

    def add_targets(self) -> "FeatureEngineer":
        """
        Create all target variables. Uses shift(-horizon) — always the LAST
        step before dropping NaN rows (never shift before feature creation).

        Targets:
          Target_Price  : next-day closing price (regression)
          Target_Return : next-day log return    (regression)
          Target_Dir    : 1=up, 0=down           (classification)
        """
        horizon = self._feat_cfg.get("prediction_horizon", 1)

        self.df["Target_Price"]  = self.df["Close"].shift(-horizon)
        self.df["Target_Return"] = self.df["LogReturn"].shift(-horizon)
        self.df["Target_Dir"]    = (self.df["Target_Return"] > 0).astype(float)

        # Multi-step targets (useful for future extensions)
        for h in [3, 5]:
            self.df[f"Target_Dir_{h}d"] = (
                self.df["Close"].shift(-h) > self.df["Close"]
            ).astype(float)

        logger.debug("✔ Target variables added")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # Main Build Method
    # ══════════════════════════════════════════════════════════════════════════

    @timer
    def build(self, drop_na: bool = True) -> pd.DataFrame:
        """
        Run all feature engineering steps in the correct order and
        return a clean feature matrix.

        Order matters:
          - ATR must be computed before Keltner Channels
          - LogReturn must exist before rolling volatility
          - Targets must be the LAST thing added (shift(-1))
          - NaN rows dropped AFTER targets to preserve alignment
        """
        logger.info(f"Building features for {len(self.df)} rows...")

        (
            self
            .add_moving_averages()
            .add_macd()
            .add_rsi()
            .add_stochastic()
            .add_roc()
            .add_williams_r()
            .add_bollinger_bands()
            .add_atr()
            .add_rolling_volatility()
            .add_keltner_channels()
            .add_volume_features()
            .add_lag_features()
            .add_rolling_stats()
            .add_calendar_features()
            .add_targets()           # ← MUST be last before dropna
        )

        n_before = len(self.df)
        if drop_na:
            self.df.dropna(inplace=True)
        n_after = len(self.df)

        feature_cols = self.get_feature_cols(self.df)
        logger.success(
            f"Feature matrix ready: {n_after} rows × {len(feature_cols)} features "
            f"(dropped {n_before - n_after} NaN rows)"
        )
        return self.df

    # ══════════════════════════════════════════════════════════════════════════
    # Helper Methods
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def get_feature_cols(df: pd.DataFrame) -> List[str]:
        """
        Return only the engineered feature columns —
        excludes raw OHLCV, LogReturn, and target columns.
        These are the columns to pass to your model as X.
        """
        exclude = {
            "Open", "High", "Low", "Close", "Volume",
            "LogReturn",
            "Target_Price", "Target_Return", "Target_Dir",
            "Target_Dir_3d", "Target_Dir_5d",
        }
        return [c for c in df.columns if c not in exclude]

    @staticmethod
    def get_X_y(
        df: pd.DataFrame,
        target: str = "Target_Dir",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split feature matrix into X (features) and y (target).

        Parameters
        ----------
        df     : Output of fe.build()
        target : One of 'Target_Dir', 'Target_Return', 'Target_Price'

        Returns
        -------
        X : pd.DataFrame  — feature columns only
        y : pd.Series     — target column
        """
        if target not in df.columns:
            raise ValueError(
                f"Target '{target}' not found. "
                f"Available: {[c for c in df.columns if 'Target' in c]}"
            )
        feature_cols = FeatureEngineer.get_feature_cols(df)
        X = df[feature_cols]
        y = df[target]
        return X, y

    def feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Print a summary table of all engineered features:
        name, null count, mean, std, min, max.
        Useful for sanity-checking after build().
        """
        feature_cols = self.get_feature_cols(df)
        summary = df[feature_cols].describe().T
        summary["nulls"] = df[feature_cols].isnull().sum()
        summary = summary[["nulls", "mean", "std", "min", "max"]]
        print(f"\n{'─'*65}")
        print(f"  Feature Summary  ({len(feature_cols)} features)")
        print(f"{'─'*65}")
        print(summary.to_string())
        print(f"{'─'*65}\n")
        return summary