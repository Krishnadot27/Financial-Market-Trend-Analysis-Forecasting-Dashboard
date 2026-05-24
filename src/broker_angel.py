"""
src/broker_angel.py
────────────────────
Angel One SmartAPI broker integration — READ-ONLY holdings fetch.

Only reads holdings/positions. No trade execution ever.

Setup (one-time):
  1. Create account at angelbroking.com
  2. Go to smartapi.angelbroking.com → Create App → get API Key
  3. Enable TOTP on your Angel One account (Google Authenticator)
  4. Add to config.yaml:
       broker:
         angel_one:
           enabled:   true
           api_key:   "your_api_key"
           client_id: "your_client_id"   # Angel One login ID
           password:  "your_password"    # Angel One PIN
           totp_key:  "your_totp_secret" # 32-char TOTP secret from Angel One

  5. pip install smartapi-python pyotp

Usage:
    from src.broker_angel import AngelOneBroker

    broker   = AngelOneBroker(config)
    holdings = broker.get_holdings()   # list of holding dicts
    profile  = broker.get_profile()    # name, account info
"""

from __future__ import annotations

import os
from typing import Optional
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────

class AngelOneBroker:
    """
    Read-only Angel One SmartAPI client.
    Fetches holdings and enriches with current prices + ML predictions.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("broker", {}).get("angel_one", {})
        self.enabled    = cfg.get("enabled",   False)
        self.api_key    = cfg.get("api_key",   os.getenv("ANGEL_API_KEY",    ""))
        self.client_id  = cfg.get("client_id", os.getenv("ANGEL_CLIENT_ID",  ""))
        self.password   = cfg.get("password",  os.getenv("ANGEL_PASSWORD",   ""))
        self.totp_key   = cfg.get("totp_key",  os.getenv("ANGEL_TOTP_KEY",   ""))
        self._session   = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Authenticate with Angel One SmartAPI.
        Returns True if successful, False otherwise.
        """
        if not self.enabled:
            logger.info("Angel One broker disabled in config")
            return False

        if not all([self.api_key, self.client_id, self.password]):
            logger.warning(
                "Angel One credentials missing. Add to config.yaml:\n"
                "  broker:\n"
                "    angel_one:\n"
                "      enabled:   true\n"
                "      api_key:   your_api_key\n"
                "      client_id: your_client_id\n"
                "      password:  your_password\n"
                "      totp_key:  your_totp_secret"
            )
            return False

        try:
            from SmartApi import SmartConnect
            import pyotp
        except ImportError:
            logger.error(
                "Missing packages. Install:\n"
                "  pip install smartapi-python pyotp"
            )
            return False

        try:
            totp = pyotp.TOTP(self.totp_key).now() if self.totp_key else ""
            self._session = SmartConnect(api_key=self.api_key)
            data = self._session.generateSession(
                self.client_id, self.password, totp
            )
            if data["status"]:
                logger.success(
                    f"Angel One connected: {data['data'].get('name', self.client_id)}"
                )
                return True
            else:
                logger.error(f"Angel One auth failed: {data.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Angel One connection error: {e}")
            return False

    def is_connected(self) -> bool:
        return self._session is not None

    # ── Holdings ──────────────────────────────────────────────────────────────

    def get_holdings(self) -> list[dict]:
        """
        Fetch all holdings from Angel One.

        Returns list of dicts with:
            ticker, tradingsymbol, isin, qty, avg_price,
            ltp (last traded price), pnl, pnl_pct, current_value
        """
        if not self.is_connected():
            if not self.connect():
                return []
        try:
            resp     = self._session.holding()
            raw_list = resp.get("data", []) or []
            holdings = []
            for h in raw_list:
                qty       = float(h.get("quantity",        0))
                avg_price = float(h.get("averageprice",    0))
                ltp       = float(h.get("ltp",             0))
                symbol    = h.get("tradingsymbol", "")
                ticker    = symbol + ".NS" if not symbol.endswith(".NS") else symbol
                cur_val   = round(qty * ltp,       2)
                cost      = round(qty * avg_price, 2)
                pnl       = round(cur_val - cost,  2)
                pnl_pct   = round((pnl / cost * 100) if cost > 0 else 0, 2)
                holdings.append({
                    "ticker":        ticker,
                    "tradingsymbol": symbol,
                    "isin":          h.get("isin", ""),
                    "qty":           qty,
                    "avg_price":     avg_price,
                    "ltp":           ltp,
                    "current_value": cur_val,
                    "cost_value":    cost,
                    "pnl":           pnl,
                    "pnl_pct":       pnl_pct,
                    "exchange":      h.get("exchange", "NSE"),
                    "product":       h.get("product",  ""),
                })
            logger.success(f"Fetched {len(holdings)} holdings from Angel One")
            return holdings
        except Exception as e:
            logger.error(f"Failed to fetch holdings: {e}")
            return []

    def get_profile(self) -> dict:
        """Fetch account profile — name, client ID, email."""
        if not self.is_connected():
            if not self.connect():
                return {}
        try:
            resp = self._session.getProfile(
                self._session.generateToken()["data"]["refreshToken"]
            )
            data = resp.get("data", {})
            return {
                "name":      data.get("name",      self.client_id),
                "client_id": data.get("clientcode", self.client_id),
                "email":     data.get("email",     ""),
                "broker":    "Angel One",
            }
        except Exception as e:
            logger.warning(f"Could not fetch profile: {e}")
            return {"name": self.client_id, "broker": "Angel One"}


# ─────────────────────────────────────────────────────────────────────────────
# Manual holdings fallback
# ─────────────────────────────────────────────────────────────────────────────

class ManualHoldingsManager:
    """
    Fallback for users without broker API.
    Holdings entered manually and stored in SQLite.
    """

    def __init__(self, db_path=None) -> None:
        import sqlite3
        from src.utils import project_path, ensure_dirs
        self.db_path = Path(db_path) if db_path else project_path("data", "predictions.db")
        ensure_dirs(self.db_path.parent)
        self._init_table()

    def _conn(self):
        import sqlite3
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_holdings (
                    ticker      TEXT PRIMARY KEY,
                    qty         REAL NOT NULL,
                    avg_price   REAL NOT NULL,
                    added_at    TEXT NOT NULL,
                    notes       TEXT
                )
            """)

    def add(self, ticker: str, qty: float, avg_price: float, notes: str = "") -> None:
        ticker = ticker.upper().strip()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO manual_holdings VALUES (?,?,?,?,?)",
                (ticker, qty, avg_price,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notes),
            )
        logger.success(f"Added holding: {ticker} qty={qty} avg=₹{avg_price}")

    def remove(self, ticker: str) -> None:
        ticker = ticker.upper().strip()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        with self._conn() as conn:
            conn.execute("DELETE FROM manual_holdings WHERE ticker=?", (ticker,))

    def get_all_raw(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ticker,qty,avg_price,notes FROM manual_holdings"
            ).fetchall()
        return [{"ticker": r[0], "qty": r[1], "avg_price": r[2], "notes": r[3]}
                for r in rows]

    def get_holdings(self) -> list[dict]:
        """Fetch stored holdings enriched with current prices from yfinance."""
        import yfinance as yf
        raw = self.get_all_raw()
        if not raw:
            return []
        holdings = []
        for h in raw:
            ticker    = h["ticker"]
            qty       = h["qty"]
            avg_price = h["avg_price"]
            try:
                df  = yf.download(ticker, period="5d",
                                  auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                ltp = float(df["Close"].iloc[-1]) if len(df) > 0 else avg_price
            except Exception:
                ltp = avg_price
            cur_val = round(qty * ltp,       2)
            cost    = round(qty * avg_price, 2)
            pnl     = round(cur_val - cost,  2)
            pnl_pct = round((pnl / cost * 100) if cost > 0 else 0, 2)
            holdings.append({
                "ticker":        ticker,
                "tradingsymbol": ticker.replace(".NS", ""),
                "qty":           qty,
                "avg_price":     avg_price,
                "ltp":           ltp,
                "current_value": cur_val,
                "cost_value":    cost,
                "pnl":           pnl,
                "pnl_pct":       pnl_pct,
                "exchange":      "NSE",
                "product":       "Manual",
            })
        return holdings


def get_broker(config: dict):
    """
    Factory — returns the right broker/holdings object based on config.
    Falls back to manual if Angel One not configured.
    """
    angel_cfg = config.get("broker", {}).get("angel_one", {})
    if angel_cfg.get("enabled") and angel_cfg.get("api_key"):
        broker = AngelOneBroker(config)
        if broker.connect():
            return broker, "angel"
    return ManualHoldingsManager(), "manual"


from datetime import datetime
from pathlib import Path
import pandas as pd