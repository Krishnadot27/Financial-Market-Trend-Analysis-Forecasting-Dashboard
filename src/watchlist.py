"""
src/watchlist.py
─────────────────
Personal stock watchlist — up to 20 NSE stocks.

Stores watchlist in SQLite alongside predictions.db.
Fetches live prices, computes signals, returns enriched data
for the dashboard.

Usage:
    from src.watchlist import WatchlistManager

    wm = WatchlistManager()
    wm.add("RELIANCE.NS")
    wm.add("INFY.NS")
    stocks = wm.fetch_all()     # list of enriched dicts
    wm.remove("INFY.NS")
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import project_path, ensure_dirs

DB_PATH  = project_path("data", "predictions.db")
MAX_SIZE = 20


# ─────────────────────────────────────────────────────────────────────────────

class WatchlistManager:
    """Manages a personal stock watchlist persisted in SQLite."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DB_PATH)
        ensure_dirs(self.db_path.parent)
        self._init_table()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_table(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    ticker     TEXT PRIMARY KEY,
                    added_at   TEXT NOT NULL,
                    notes      TEXT
                )
            """)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, ticker: str, notes: str = "") -> bool:
        ticker = ticker.upper().strip()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        current = self.get_tickers()
        if ticker in current:
            logger.info(f"{ticker} already in watchlist")
            return False
        if len(current) >= MAX_SIZE:
            logger.warning(f"Watchlist full ({MAX_SIZE} stocks max)")
            return False
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO watchlist VALUES (?,?,?)",
                (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notes),
            )
        logger.success(f"Added {ticker} to watchlist")
        return True

    def remove(self, ticker: str) -> bool:
        ticker = ticker.upper().strip()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        with self._conn() as conn:
            n = conn.execute(
                "DELETE FROM watchlist WHERE ticker=?", (ticker,)
            ).rowcount
        return n > 0

    def get_tickers(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ticker FROM watchlist ORDER BY added_at"
            ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        return len(self.get_tickers())

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM watchlist")

    # ── Live data fetch ───────────────────────────────────────────────────────

    def fetch_all(self, config: dict = None) -> list[dict]:
        """
        Fetch live price + signal for all watchlist stocks.

        Returns list of dicts, one per ticker, with:
            ticker, name, price, change_pct, change_abs,
            high_52w, low_52w, volume, volume_ratio,
            direction, confidence, regime,
            sparkline (list of 30 closes)
        """
        import yfinance as yf
        tickers = self.get_tickers()
        if not tickers:
            return []

        results = []
        for ticker in tickers:
            try:
                result = self._fetch_one(ticker, config or {})
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Watchlist fetch failed for {ticker}: {e}")
                results.append({
                    "ticker": ticker,
                    "name":   ticker.replace(".NS", ""),
                    "error":  str(e),
                    "price":  None,
                })
        return results

    def _fetch_one(self, ticker: str, config: dict) -> dict:
        import yfinance as yf
        from src.feature_engineering import FeatureEngineer
        from src.regime_detection import RegimeDetector

        # Fetch 1 year daily data
        df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if len(df) < 50:
            return None

        price     = float(df["Close"].iloc[-1])
        prev      = float(df["Close"].iloc[-2]) if len(df) > 1 else price
        chg_abs   = round(price - prev, 2)
        chg_pct   = round((price - prev) / prev * 100, 2) if prev else 0
        high_52w  = round(float(df["High"].max()), 2)
        low_52w   = round(float(df["Low"].min()),  2)
        volume    = int(df["Volume"].iloc[-1])
        vol_avg   = float(df["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = round(volume / vol_avg, 2) if vol_avg else 1.0

        # Sparkline: last 30 closes normalised 0-100
        closes    = df["Close"].iloc[-30:].values.tolist()

        # Regime
        regime_info = {"regime": "UNKNOWN", "color": "#64748B", "emoji": "?"}
        try:
            fe  = FeatureEngineer(df, config=config)
            df_feat = fe.build()
            rd  = RegimeDetector()
            regime_info = rd.detect(df_feat)
        except Exception:
            pass

        # ML signal from prediction store (last stored prediction)
        direction  = None
        confidence = None
        try:
            from src.prediction_store import PredictionStore
            store   = PredictionStore(self.db_path)
            hist    = store.get_history(ticker=ticker, timeframe="daily", limit=1)
            if len(hist) > 0:
                direction  = hist.iloc[0].get("direction")
                confidence = hist.iloc[0].get("confidence")
        except Exception:
            pass

        # Info from yfinance
        try:
            info = yf.Ticker(ticker).fast_info
            name = getattr(info, "long_name",
                   getattr(info, "shortName", ticker.replace(".NS", "")))
            if not name or name == ticker:
                name = ticker.replace(".NS", "")
        except Exception:
            name = ticker.replace(".NS", "")

        return {
            "ticker":     ticker,
            "name":       str(name)[:25],
            "price":      price,
            "change_pct": chg_pct,
            "change_abs": chg_abs,
            "high_52w":   high_52w,
            "low_52w":    low_52w,
            "volume":     volume,
            "volume_ratio": vol_ratio,
            "direction":  direction,
            "confidence": confidence,
            "regime":     regime_info.get("regime", "UNKNOWN"),
            "regime_color": regime_info.get("color", "#64748B"),
            "regime_emoji": regime_info.get("emoji", ""),
            "sparkline":  closes,
            "error":      None,
        }