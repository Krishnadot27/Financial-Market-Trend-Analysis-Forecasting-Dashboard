"""
src/intraday.py
────────────────
Multi-timeframe intraday prediction engine for NSE stocks.

Timeframes: 5min, 15min, 1hr
Predicts:   Direction (Up/Down) + Price targets (Entry, Target, Stop Loss)
Alerts:     Only when signal confidence > threshold (strong signals only)

Key concepts:
  - Downloads intraday OHLCV via yfinance (free, no API key)
  - Builds intraday-specific features (VWAP, intraday momentum, etc.)
  - Uses trained XGBoost model adapted for intraday
  - Computes price targets using ATR-based risk/reward
  - NSE market hours: 9:15 AM – 3:30 PM IST

Usage:
    from src.intraday import IntradayPredictor

    ip = IntradayPredictor(ticker="RELIANCE.NS", config=cfg)
    result = ip.predict_all_timeframes()
    ip.print_report(result)
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

NSE_OPEN  = time(9, 15)
NSE_CLOSE = time(15, 30)

TIMEFRAME_MAP = {
    "5min":  "5m",
    "15min": "15m",
    "1hr":   "60m",
}

# yfinance period for each interval
PERIOD_MAP = {
    "5min":  "5d",
    "15min": "5d",
    "1hr":   "30d",
}

# Minimum candles needed before prediction
MIN_CANDLES = {
    "5min":  50,
    "15min": 30,
    "1hr":   20,
}

# Signal strength thresholds
STRONG_SIGNAL_THRESHOLD = 0.65   # confidence > 65% = strong signal
VERY_STRONG_THRESHOLD   = 0.80   # confidence > 80% = very strong


# ══════════════════════════════════════════════════════════════════════════════
# Intraday feature engineering
# ══════════════════════════════════════════════════════════════════════════════

class IntradayFeatureEngineer:
    """
    Builds intraday-specific features from OHLCV data.

    Features added on top of standard technical indicators:
      - VWAP (Volume Weighted Average Price)
      - VWAP deviation (price vs VWAP)
      - Intraday momentum (first 30min range)
      - Time-of-day features (opening hour, closing hour, mid-session)
      - Previous day high/low reference levels
      - Gap up/down from previous close
      - Intraday cumulative return
      - Rolling session high/low
    """

    def __init__(self, df: pd.DataFrame, timeframe: str = "15min") -> None:
        self.df        = df.copy()
        self.timeframe = timeframe

    def build(self) -> pd.DataFrame:
        """Build all intraday features. Returns enriched DataFrame."""
        df = self.df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # ── Standard price features ────────────────────────────────────────────
        df["Returns"]    = df["Close"].pct_change().fillna(0)
        df["LogReturn"]  = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)
        df["HL_Range"]   = (df["High"] - df["Low"]) / df["Close"]
        df["Body_Size"]  = abs(df["Close"] - df["Open"]) / df["Close"]
        df["Upper_Wick"] = (df["High"] - df[["Close","Open"]].max(axis=1)) / df["Close"]
        df["Lower_Wick"] = (df[["Close","Open"]].min(axis=1) - df["Low"]) / df["Close"]

        # ── VWAP ──────────────────────────────────────────────────────────────
        typical        = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"]     = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["VWAP_dev"] = (df["Close"] - df["VWAP"]) / df["VWAP"]

        # ── Moving averages (short windows for intraday) ───────────────────────
        for w in [5, 10, 20]:
            df[f"SMA_{w}"]    = df["Close"].rolling(w).mean()
            df[f"EMA_{w}"]    = df["Close"].ewm(span=w).mean()
            df[f"Price_SMA{w}_ratio"] = df["Close"] / df[f"SMA_{w}"].replace(0, np.nan) - 1

        # ── RSI ───────────────────────────────────────────────────────────────
        for period in [7, 14]:
            delta   = df["Close"].diff()
            gain    = delta.clip(lower=0).rolling(period).mean()
            loss    = (-delta.clip(upper=0)).rolling(period).mean()
            rs      = gain / loss.replace(0, np.nan)
            df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

        # ── Bollinger Bands ────────────────────────────────────────────────────
        mid            = df["Close"].rolling(20).mean()
        std            = df["Close"].rolling(20).std()
        df["BB_Upper"] = mid + 2 * std
        df["BB_Lower"] = mid - 2 * std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid.replace(0, np.nan)
        df["BB_PctB"]  = (df["Close"] - df["BB_Lower"]) / (
            df["BB_Upper"] - df["BB_Lower"]
        ).replace(0, np.nan)

        # ── ATR ───────────────────────────────────────────────────────────────
        hl   = df["High"] - df["Low"]
        hc   = (df["High"] - df["Close"].shift(1)).abs()
        lc   = (df["Low"]  - df["Close"].shift(1)).abs()
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["ATR"]     = tr.rolling(14).mean()
        df["ATR_pct"] = df["ATR"] / df["Close"]

        # ── Volume features ────────────────────────────────────────────────────
        df["Volume_SMA20"]  = df["Volume"].rolling(20).mean()
        df["Volume_ratio"]  = df["Volume"] / df["Volume_SMA20"].replace(0, np.nan)
        df["Volume_surge"]  = (df["Volume_ratio"] > 2.0).astype(int)

        # ── Momentum ──────────────────────────────────────────────────────────
        for period in [3, 5, 10]:
            df[f"ROC_{period}"] = df["Close"].pct_change(period)
            df[f"Lag_{period}"] = df["Close"].shift(period) / df["Close"] - 1

        # ── MACD ──────────────────────────────────────────────────────────────
        ema12          = df["Close"].ewm(span=12).mean()
        ema26          = df["Close"].ewm(span=26).mean()
        df["MACD"]     = ema12 - ema26
        df["MACD_Sig"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Hist"]= df["MACD"] - df["MACD_Sig"]

        # ── Stochastic ────────────────────────────────────────────────────────
        low14   = df["Low"].rolling(14).min()
        high14  = df["High"].rolling(14).max()
        df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14).replace(0, np.nan)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

        # ── Time-of-day features ──────────────────────────────────────────────
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        except Exception:
            pass

        if isinstance(df.index, pd.DatetimeIndex):
            df["Hour"]          = df.index.hour
            df["Minute"]        = df.index.minute
            df["IsOpeningHour"] = (df.index.hour == 9).astype(int)
            df["IsClosingHour"] = (df.index.hour == 15).astype(int)
            df["IsMidSession"]  = (
                (df.index.hour >= 11) & (df.index.hour <= 13)
            ).astype(int)
            minutes_from_open      = (df.index.hour - 9) * 60 + df.index.minute - 15
            minutes_series         = pd.Series(minutes_from_open, index=df.index, dtype=float)
            df["Session_progress"] = minutes_series.clip(lower=0, upper=375) / 375
        else:
            df["Hour"]             = 12
            df["Minute"]           = 0
            df["IsOpeningHour"]    = 0
            df["IsClosingHour"]    = 0
            df["IsMidSession"]     = 1
            df["Session_progress"] = 0.5

        # ── Session rolling high/low ───────────────────────────────────────────
        df["Session_High"] = df["High"].rolling(20, min_periods=1).max()
        df["Session_Low"]  = df["Low"].rolling(20, min_periods=1).min()
        df["Pct_from_High"] = (df["Close"] - df["Session_High"]) / df["Session_High"]
        df["Pct_from_Low"]  = (df["Close"] - df["Session_Low"]) / df["Session_Low"]

        # ── Gap from previous candle ───────────────────────────────────────────
        df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

        # ── Target: next candle direction ──────────────────────────────────────
        df["Target_Dir"]    = (df["Close"].shift(-1) > df["Close"]).astype(float)
        df["Target_Return"] = df["Close"].shift(-1) / df["Close"] - 1

        df = df.dropna()
        return df

    @staticmethod
    def get_feature_cols(df: pd.DataFrame) -> list[str]:
        """Return all feature column names (excludes OHLCV + targets)."""
        exclude = {"Open","High","Low","Close","Volume",
                   "Target_Dir","Target_Return","VWAP"}
        return [c for c in df.columns if c not in exclude]


# ══════════════════════════════════════════════════════════════════════════════
# Price target calculator
# ══════════════════════════════════════════════════════════════════════════════

class PriceTargetCalculator:
    """
    Computes entry, target, and stop-loss levels using ATR.

    Risk/Reward ratios:
      Conservative: 1:1.5
      Moderate:     1:2.0
      Aggressive:   1:3.0
    """

    def __init__(self, atr_multiplier_sl: float = 1.5,
                 risk_reward: float = 2.0) -> None:
        self.atr_multiplier_sl = atr_multiplier_sl
        self.risk_reward       = risk_reward

    def compute(
        self,
        current_price: float,
        atr:           float,
        direction:     str,
        confidence:    float = 0.7,
    ) -> dict:
        """
        Compute entry, target, stop-loss levels.

        Parameters
        ----------
        current_price : Current close price
        atr           : Average True Range (in price units)
        direction     : 'UP' or 'DOWN'
        confidence    : Model confidence (0-1)

        Returns
        -------
        dict with entry, target, stop_loss, risk, reward, rr_ratio
        """
        # Adjust ATR multiplier by confidence
        sl_mult = self.atr_multiplier_sl * (1 + (1 - confidence) * 0.5)
        rr      = self.risk_reward * confidence + 1.0 * (1 - confidence)

        if direction == "UP":
            entry     = current_price
            stop_loss = round(entry - (atr * sl_mult), 2)
            target    = round(entry + (atr * sl_mult * rr), 2)
        else:  # DOWN
            entry     = current_price
            stop_loss = round(entry + (atr * sl_mult), 2)
            target    = round(entry - (atr * sl_mult * rr), 2)

        risk   = abs(entry - stop_loss)
        reward = abs(target - entry)

        return {
            "entry":      round(entry,     2),
            "target":     round(target,    2),
            "stop_loss":  round(stop_loss, 2),
            "risk":       round(risk,      2),
            "reward":     round(reward,    2),
            "rr_ratio":   round(reward / risk, 2) if risk > 0 else 0,
            "risk_pct":   round(risk / entry * 100, 2),
            "reward_pct": round(reward / entry * 100, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Intraday Predictor
# ══════════════════════════════════════════════════════════════════════════════

class IntradayPredictor:
    """
    Multi-timeframe intraday predictor.

    Uses the trained daily XGBoost model as a base, adapts features
    for intraday timeframes, and computes price targets.

    Parameters
    ----------
    ticker   : NSE ticker e.g. 'RELIANCE.NS'
    config   : loaded config dict
    model    : optional pre-loaded model (loads from disk if None)
    """

    def __init__(
        self,
        ticker:  str,
        config:  dict,
        model    = None,
    ) -> None:
        self.ticker = ticker
        self.config = config
        self.model  = model
        self.ptc    = PriceTargetCalculator(
            atr_multiplier_sl = config.get("intraday", {}).get("atr_sl_mult", 1.5),
            risk_reward       = config.get("intraday", {}).get("risk_reward",  2.0),
        )

    def _load_model(self, task: str = "regression"):
        """Load model from disk."""
        import joblib
        from src.utils import project_path

        safe  = self.ticker.replace(".", "_")
        path  = project_path("models", f"xgboost_{task}_{safe}.pkl")
        if not path.exists():
            logger.warning(f"No model found at {path}")
            return None
        obj = joblib.load(path)
        return obj.get("pipeline") if isinstance(obj, dict) else obj

    def _fetch_intraday(self, timeframe: str) -> pd.DataFrame:
        """Fetch intraday OHLCV from yfinance."""
        import yfinance as yf

        interval = TIMEFRAME_MAP[timeframe]
        period   = PERIOD_MAP[timeframe]

        df = yf.download(
            self.ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df.index = pd.to_datetime(df.index)

        # Convert to IST if timezone-aware
        if df.index.tz is not None:
            try:
                df.index = df.index.tz_convert("Asia/Kolkata")
            except Exception:
                df.index = df.index.tz_localize(None)
        df.index.name = "Datetime"

        # Filter to NSE market hours only
        df = df[
            (df.index.time >= NSE_OPEN) &
            (df.index.time <= NSE_CLOSE)
        ].copy()

        return df.dropna()

    def predict_timeframe(self, timeframe: str) -> dict:
        """
        Generate prediction for a single timeframe.

        Returns dict with direction, confidence, price targets,
        signal strength, and raw data.
        """
        logger.info(f"Predicting {self.ticker} [{timeframe}]...")

        try:
            # Fetch data
            df_raw = self._fetch_intraday(timeframe)
            if len(df_raw) < MIN_CANDLES[timeframe]:
                return {
                    "timeframe":  timeframe,
                    "status":     "insufficient_data",
                    "error":      f"Need {MIN_CANDLES[timeframe]} candles, got {len(df_raw)}",
                }

            # Build features
            fe     = IntradayFeatureEngineer(df_raw, timeframe)
            df_feat = fe.build()

            if len(df_feat) < 10:
                return {"timeframe": timeframe, "status": "feature_build_failed"}

            feature_cols  = fe.get_feature_cols(df_feat)
            # Only use features that exist
            feature_cols  = [c for c in feature_cols if c in df_feat.columns]
            X_latest      = df_feat[feature_cols].iloc[[-1]]
            current_price = float(df_feat["Close"].iloc[-1])
            atr           = float(df_feat["ATR"].iloc[-1]) if "ATR" in df_feat.columns else current_price * 0.005

            # Load or use provided model
            model = self.model or self._load_model("regression")
            if model is None:
                return {"timeframe": timeframe, "status": "no_model"}

            # Predict — align features to what model expects
            try:
                # Get model's expected feature names
                estimator = model
                if hasattr(model, 'named_steps'):
                    estimator = model.named_steps.get('model', model)

                if hasattr(estimator, 'feature_names_in_'):
                    model_features = list(estimator.feature_names_in_)
                elif hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                else:
                    model_features = feature_cols

                # Build aligned row: use 0.0 for missing features
                aligned = pd.DataFrame(
                    [[0.0] * len(model_features)],
                    columns=model_features,
                )
                # Fill in features we actually have
                common = [c for c in model_features if c in X_latest.columns]
                aligned[common] = X_latest[common].values

                pred = float(model.predict(aligned)[0])
            except Exception as e:
                return {"timeframe": timeframe, "status": "prediction_failed", "error": str(e)}

            # Direction and confidence
            direction  = "UP" if pred > 0 else "DOWN"
            confidence = min(abs(pred) * 10, 1.0)   # scale to 0-1
            confidence = max(confidence, 0.5)         # minimum 50%

            # Signal strength
            if confidence >= VERY_STRONG_THRESHOLD:
                strength = "VERY STRONG"
                is_alert = True
            elif confidence >= STRONG_SIGNAL_THRESHOLD:
                strength = "STRONG"
                is_alert = True
            elif confidence >= 0.55:
                strength = "MODERATE"
                is_alert = False
            else:
                strength = "WEAK"
                is_alert = False

            # Price targets
            targets = self.ptc.compute(current_price, atr, direction, confidence)

            # Latest candle info
            last_candle = df_feat.iloc[-1]
            rsi   = float(df_feat["RSI_14"].iloc[-1]) if "RSI_14" in df_feat.columns else 50
            vwap  = float(df_feat["VWAP"].iloc[-1])   if "VWAP"   in df_feat.columns else current_price
            vol_r = float(df_feat["Volume_ratio"].iloc[-1]) if "Volume_ratio" in df_feat.columns else 1.0

            return {
                "timeframe":     timeframe,
                "status":        "ok",
                "ticker":        self.ticker,
                "timestamp":     datetime.now().strftime("%H:%M:%S IST"),
                "current_price": current_price,
                "direction":     direction,
                "confidence":    round(confidence, 4),
                "raw_score":     round(pred, 6),
                "strength":      strength,
                "is_alert":      is_alert,
                "targets":       targets,
                "indicators": {
                    "RSI":          round(rsi, 1),
                    "VWAP":         round(vwap, 2),
                    "VWAP_dev":     f"{(current_price/vwap - 1)*100:+.2f}%",
                    "Volume_ratio": round(vol_r, 2),
                    "ATR":          round(atr, 2),
                },
                "candles_used":  len(df_feat),
                "df":            df_feat,
            }

        except Exception as e:
            logger.error(f"Error predicting {self.ticker} [{timeframe}]: {e}")
            return {"timeframe": timeframe, "status": "error", "error": str(e)}

    def predict_all_timeframes(self) -> dict:
        """
        Predict across all 3 timeframes: 5min, 15min, 1hr.

        Returns combined result with confluence score.
        """
        results = {}
        for tf in ["5min", "15min", "1hr"]:
            results[tf] = self.predict_timeframe(tf)

        # Confluence: how many timeframes agree on direction
        valid     = [r for r in results.values() if r.get("status") == "ok"]
        up_count  = sum(1 for r in valid if r["direction"] == "UP")
        dn_count  = sum(1 for r in valid if r["direction"] == "DOWN")
        total     = len(valid)

        if total == 0:
            confluence = "NO DATA"
            conf_score = 0
        elif up_count == total:
            confluence = "STRONG BUY"
            conf_score = 1.0
        elif dn_count == total:
            confluence = "STRONG SELL"
            conf_score = 1.0
        elif up_count > dn_count:
            confluence = "LEAN BUY"
            conf_score = up_count / total
        elif dn_count > up_count:
            confluence = "LEAN SELL"
            conf_score = dn_count / total
        else:
            confluence = "NEUTRAL"
            conf_score = 0.5

        # Best timeframe = highest confidence
        best = max(valid, key=lambda r: r.get("confidence", 0)) if valid else None

        return {
            "ticker":      self.ticker,
            "timestamp":   datetime.now().strftime("%d %b %Y %H:%M IST"),
            "timeframes":  results,
            "confluence":  confluence,
            "conf_score":  conf_score,
            "best":        best,
            "is_market_open": self._is_market_open(),
        }

    @staticmethod
    def _is_market_open() -> bool:
        """Check if NSE is currently open."""
        now = datetime.now().time()
        return NSE_OPEN <= now <= NSE_CLOSE

    def print_report(self, result: dict) -> None:
        """Print formatted intraday prediction report."""
        print(f"\n{'═'*60}")
        print(f"  INTRADAY PREDICTION — {result['ticker']}")
        print(f"  {result['timestamp']}")
        print(f"{'─'*60}")
        print(f"  Confluence: {result['confluence']}  ({result['conf_score']:.0%})")
        print(f"  Market Open: {'Yes' if result['is_market_open'] else 'No (using last session)'}")
        print(f"{'─'*60}")

        for tf, r in result["timeframes"].items():
            if r.get("status") != "ok":
                print(f"  [{tf:>5}]  ✘ {r.get('status','error')}: {r.get('error','')[:40]}")
                continue

            d   = r["direction"]
            sym = "▲" if d == "UP" else "▼"
            col = "🟢" if d == "UP" else "🔴"
            print(f"\n  [{tf:>5}]  {col} {sym} {d}  |  "
                  f"Confidence: {r['confidence']:.0%}  |  "
                  f"Strength: {r['strength']}")
            print(f"          Price: ₹{r['current_price']:,.2f}")

            t = r["targets"]
            print(f"          Entry:     ₹{t['entry']:>10,.2f}")
            print(f"          Target:    ₹{t['target']:>10,.2f}  (+{t['reward_pct']:.2f}%)")
            print(f"          Stop Loss: ₹{t['stop_loss']:>10,.2f}  (-{t['risk_pct']:.2f}%)")
            print(f"          R:R Ratio: 1:{t['rr_ratio']:.1f}")

            ind = r.get("indicators", {})
            print(f"          RSI: {ind.get('RSI','—')}  "
                  f"VWAP: ₹{ind.get('VWAP','—')}  "
                  f"VWAP Dev: {ind.get('VWAP_dev','—')}  "
                  f"Vol: {ind.get('Volume_ratio','—')}x")

            if r["is_alert"]:
                print(f"          🔔 ALERT — {r['strength']} SIGNAL DETECTED")

        print(f"\n{'═'*60}")
        print(f"  ⚠ Not financial advice. For research only.")
        print(f"{'═'*60}\n")