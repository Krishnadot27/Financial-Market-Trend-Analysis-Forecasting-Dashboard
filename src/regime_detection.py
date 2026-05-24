"""
src/regime_detection.py
────────────────────────
Market regime detection: Bull / Bear / Sideways

Uses a combination of:
  - Trend indicators (SMA 50/200, Golden/Death Cross)
  - Volatility (ATR, Bollinger Band width)
  - Momentum (RSI, ROC)
  - Volume confirmation

Returns one of 3 regimes:
  BULL     — uptrend, low volatility, positive momentum
  BEAR     — downtrend, high volatility, negative momentum
  SIDEWAYS — range-bound, low momentum, mixed signals

Usage:
    from src.regime_detection import RegimeDetector

    rd = RegimeDetector()
    regime = rd.detect(df_features)
    # → {'regime': 'BULL', 'confidence': 0.82, 'signals': {...}}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger


class RegimeDetector:
    """
    Rule-based market regime detector using technical indicators.

    Scores each regime using a weighted voting system across
    multiple indicators. Returns the regime with highest score.
    """

    REGIMES = ("BULL", "BEAR", "SIDEWAYS")

    REGIME_COLORS = {
        "BULL":     "#22C55E",
        "BEAR":     "#EF4444",
        "SIDEWAYS": "#F59E0B",
    }

    REGIME_EMOJI = {
        "BULL":     "🐂",
        "BEAR":     "🐻",
        "SIDEWAYS": "↔️",
    }

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detect current market regime from feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix from FeatureEngineer.build()
            Must have at least 50 rows.

        Returns
        -------
        dict with keys:
            regime     : 'BULL' | 'BEAR' | 'SIDEWAYS'
            confidence : float 0-1
            score      : {'BULL': float, 'BEAR': float, 'SIDEWAYS': float}
            signals    : dict of individual indicator signals
            description: human-readable explanation
        """
        if len(df) < 50:
            return self._unknown()

        row    = df.iloc[-1]
        scores = {"BULL": 0.0, "BEAR": 0.0, "SIDEWAYS": 0.0}
        signals = {}

        # ── 1. Trend: SMA 50 vs SMA 200 (Golden/Death Cross) ─────────────────
        if "SMA_50" in df.columns and "SMA_200" in df.columns:
            sma50  = float(row.get("SMA_50",  0))
            sma200 = float(row.get("SMA_200", 0))
            close  = float(row.get("Close",   0))
            if sma50 > 0 and sma200 > 0:
                if sma50 > sma200 and close > sma50:
                    scores["BULL"]     += 2.0
                    signals["Golden Cross"] = "✔ SMA50 > SMA200"
                elif sma50 < sma200 and close < sma50:
                    scores["BEAR"]     += 2.0
                    signals["Death Cross"] = "✔ SMA50 < SMA200"
                else:
                    scores["SIDEWAYS"] += 1.0
                    signals["Cross"] = "Mixed — no clear cross"

        # ── 2. Momentum: RSI ──────────────────────────────────────────────────
        if "RSI_14" in df.columns:
            rsi = float(row.get("RSI_14", 50))
            signals["RSI_14"] = f"{rsi:.1f}"
            if rsi > 55:
                scores["BULL"]     += 1.5
            elif rsi < 45:
                scores["BEAR"]     += 1.5
            else:
                scores["SIDEWAYS"] += 1.0

        # ── 3. MACD ────────────────────────────────────────────────────────────
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            macd   = float(row.get("MACD",        0))
            signal = float(row.get("MACD_Signal", 0))
            signals["MACD"] = f"{macd:.3f} vs Signal {signal:.3f}"
            if macd > signal and macd > 0:
                scores["BULL"] += 1.5
            elif macd < signal and macd < 0:
                scores["BEAR"] += 1.5
            else:
                scores["SIDEWAYS"] += 0.5

        # ── 4. Bollinger Band width (volatility) ───────────────────────────────
        if "BB_Width" in df.columns:
            bb_width     = float(row.get("BB_Width", 0))
            bb_width_20d = df["BB_Width"].rolling(20).mean().iloc[-1]
            signals["BB_Width"] = f"{bb_width:.4f}"
            if bb_width < bb_width_20d * 0.8:
                scores["SIDEWAYS"] += 1.5   # squeeze = sideways
            elif bb_width > bb_width_20d * 1.5:
                # High volatility — check direction
                if scores["BULL"] > scores["BEAR"]:
                    scores["BULL"] += 0.5
                else:
                    scores["BEAR"] += 0.5

        # ── 5. Price vs SMA 20 ────────────────────────────────────────────────
        if "SMA_20" in df.columns:
            sma20 = float(row.get("SMA_20", 0))
            close = float(row.get("Close",  0))
            if sma20 > 0:
                pct_above = (close - sma20) / sma20 * 100
                signals["Price vs SMA20"] = f"{pct_above:+.2f}%"
                if pct_above > 2:
                    scores["BULL"]     += 1.0
                elif pct_above < -2:
                    scores["BEAR"]     += 1.0
                else:
                    scores["SIDEWAYS"] += 1.0

        # ── 6. ROC (Rate of Change) ────────────────────────────────────────────
        if "ROC_20" in df.columns:
            roc = float(row.get("ROC_20", 0))
            signals["ROC_20"] = f"{roc:.2f}%"
            if roc > 5:
                scores["BULL"] += 1.0
            elif roc < -5:
                scores["BEAR"] += 1.0
            else:
                scores["SIDEWAYS"] += 0.5

        # ── 7. Volume confirmation ─────────────────────────────────────────────
        if "Volume_ratio" in df.columns:
            vol_ratio = float(row.get("Volume_ratio", 1))
            signals["Volume Ratio"] = f"{vol_ratio:.2f}x"
            if vol_ratio > 1.5:
                # High volume confirms the dominant trend
                dominant = max(scores, key=scores.get)
                scores[dominant] += 0.5

        # ── Determine regime ───────────────────────────────────────────────────
        total      = sum(scores.values())
        if total == 0:
            return self._unknown()

        # Normalise to probabilities
        probs      = {k: v / total for k, v in scores.items()}
        regime     = max(probs, key=probs.get)
        confidence = probs[regime]

        description = self._describe(regime, confidence, signals)

        logger.info(
            f"Regime: {regime} (confidence: {confidence:.1%})  |  "
            f"Scores: BULL={scores['BULL']:.1f} "
            f"BEAR={scores['BEAR']:.1f} "
            f"SIDEWAYS={scores['SIDEWAYS']:.1f}"
        )

        return {
            "regime":      regime,
            "confidence":  round(confidence, 4),
            "score":       {k: round(v, 3) for k, v in probs.items()},
            "signals":     signals,
            "description": description,
            "color":       self.REGIME_COLORS[regime],
            "emoji":       self.REGIME_EMOJI[regime],
        }

    def detect_history(self, df: pd.DataFrame, window: int = 1) -> pd.Series:
        """
        Compute rolling regime over the full history.
        Returns a Series of regime labels indexed by date.
        Useful for plotting regime changes over time.
        """
        regimes = []
        for i in range(len(df)):
            if i < 50:
                regimes.append("UNKNOWN")
                continue
            slice_df = df.iloc[max(0, i-200):i+1]
            result   = self.detect(slice_df)
            regimes.append(result["regime"])

        return pd.Series(regimes, index=df.index, name="regime")

    def _describe(self, regime: str, confidence: float, signals: dict) -> str:
        conf_word = "strongly" if confidence > 0.6 else "moderately" if confidence > 0.4 else "weakly"
        if regime == "BULL":
            return (
                f"Market is {conf_word} bullish ({confidence:.0%} confidence). "
                f"Price is trending up with positive momentum. "
                f"Favorable conditions for long positions."
            )
        elif regime == "BEAR":
            return (
                f"Market is {conf_word} bearish ({confidence:.0%} confidence). "
                f"Price is trending down with negative momentum. "
                f"Consider reducing exposure or staying in cash."
            )
        else:
            return (
                f"Market is {conf_word} sideways ({confidence:.0%} confidence). "
                f"No clear trend detected. "
                f"Range-bound conditions — wait for breakout confirmation."
            )

    def _unknown(self) -> dict:
        return {
            "regime":      "UNKNOWN",
            "confidence":  0.0,
            "score":       {"BULL": 0, "BEAR": 0, "SIDEWAYS": 0},
            "signals":     {},
            "description": "Insufficient data to determine regime.",
            "color":       "#64748B",
            "emoji":       "❓",
        }