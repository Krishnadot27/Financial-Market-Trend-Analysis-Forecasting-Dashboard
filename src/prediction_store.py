"""
src/prediction_store.py
────────────────────────
Persistent prediction storage using SQLite + CSV export.

Every time a prediction is made (daily or intraday), it is saved
to a local SQLite database. The next day, the actual price is
fetched and the prediction is marked correct or incorrect.

Database schema:
  predictions table:
    id              INTEGER PRIMARY KEY
    timestamp       TEXT    when prediction was made
    date            TEXT    trading date (YYYY-MM-DD)
    ticker          TEXT    e.g. RELIANCE.NS
    timeframe       TEXT    daily / 5min / 15min / 1hr
    model_version   TEXT    v1 / v2
    task            TEXT    classification / regression
    direction       TEXT    UP / DOWN
    confidence      REAL    0.0 – 1.0
    raw_score       REAL    model output
    price_at_pred   REAL    close price when prediction was made
    target_price    REAL    predicted target price
    stop_loss       REAL    stop loss level
    regime          TEXT    BULL / BEAR / SIDEWAYS
    actual_price    REAL    actual next-day close (filled later)
    actual_direction TEXT   UP / DOWN (filled later)
    correct         INTEGER 1=correct 0=wrong NULL=pending

Usage:
    from src.prediction_store import PredictionStore

    store = PredictionStore()
    store.save(prediction_dict)          # save a prediction
    store.update_outcomes()              # fill in actuals for past predictions
    df = store.get_history()             # get all predictions as DataFrame
    store.export_csv("predictions.csv")  # export to CSV
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import project_path, ensure_dirs


# ══════════════════════════════════════════════════════════════════════════════
# Database setup
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = project_path("data", "predictions.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    date             TEXT    NOT NULL,
    ticker           TEXT    NOT NULL,
    timeframe        TEXT    NOT NULL DEFAULT 'daily',
    model_version    TEXT    NOT NULL DEFAULT 'v1',
    task             TEXT    NOT NULL DEFAULT 'classification',
    direction        TEXT    NOT NULL,
    confidence       REAL,
    raw_score        REAL,
    price_at_pred    REAL,
    target_price     REAL,
    stop_loss        REAL,
    regime           TEXT,
    actual_price     REAL,
    actual_direction TEXT,
    correct          INTEGER,
    notes            TEXT
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ticker   ON predictions(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_date     ON predictions(date);",
    "CREATE INDEX IF NOT EXISTS idx_correct  ON predictions(correct);",
    "CREATE INDEX IF NOT EXISTS idx_timeframe ON predictions(timeframe);",
]


# ══════════════════════════════════════════════════════════════════════════════
# PredictionStore
# ══════════════════════════════════════════════════════════════════════════════

class PredictionStore:
    """
    Persistent store for all ML predictions.

    Saves every prediction to SQLite and can auto-fill
    actual outcomes by fetching next-day prices from yfinance.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DB_PATH)
        ensure_dirs(self.db_path.parent)
        self._init_db()

    # ── Database init ─────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(CREATE_TABLE_SQL)
            for sql in CREATE_INDEXES_SQL:
                conn.execute(sql)
        logger.debug(f"PredictionStore ready at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Save prediction ───────────────────────────────────────────────────────

    def save(self, pred: dict) -> int:
        """
        Save a single prediction to the database.

        Parameters
        ----------
        pred : dict with keys:
            ticker         : str   e.g. "RELIANCE.NS"
            direction      : str   "UP" or "DOWN"
            confidence     : float 0-1
            raw_score      : float model output
            price_at_pred  : float current close price
            timeframe      : str   "daily" / "5min" / "15min" / "1hr"
            model_version  : str   "v1" / "v2"
            task           : str   "classification" / "regression"
            target_price   : float (optional)
            stop_loss      : float (optional)
            regime         : str   (optional)
            notes          : str   (optional)
            date           : str   YYYY-MM-DD (optional, defaults to today)

        Returns
        -------
        int: inserted row id
        """
        now  = datetime.now()
        date = pred.get("date", now.strftime("%Y-%m-%d"))

        row = (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            date,
            pred.get("ticker",        ""),
            pred.get("timeframe",     "daily"),
            pred.get("model_version", "v1"),
            pred.get("task",          "classification"),
            pred.get("direction",     ""),
            pred.get("confidence",    None),
            pred.get("raw_score",     None),
            pred.get("price_at_pred", None),
            pred.get("target_price",  None),
            pred.get("stop_loss",     None),
            pred.get("regime",        None),
            None,   # actual_price    — filled later
            None,   # actual_direction — filled later
            None,   # correct          — filled later
            pred.get("notes", None),
        )

        sql = """
            INSERT INTO predictions
            (timestamp, date, ticker, timeframe, model_version, task,
             direction, confidence, raw_score, price_at_pred,
             target_price, stop_loss, regime,
             actual_price, actual_direction, correct, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cursor = conn.execute(sql, row)
            row_id = cursor.lastrowid

        logger.debug(
            f"Saved prediction #{row_id}: {pred.get('ticker')} "
            f"{pred.get('direction')} ({pred.get('confidence', 0):.0%})"
        )
        return row_id

    def save_batch(self, predictions: list[dict]) -> int:
        """Save multiple predictions at once. Returns count saved."""
        count = 0
        for p in predictions:
            try:
                self.save(p)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save {p.get('ticker')}: {e}")
        logger.info(f"Saved {count}/{len(predictions)} predictions to store")
        return count

    # ── Update outcomes ───────────────────────────────────────────────────────

    def update_outcomes(self, days_back: int = 10) -> int:
        """
        Fetch actual prices and mark predictions correct/incorrect.

        Runs automatically — call this daily (e.g. in run_alerts.py).
        For each pending prediction older than 1 day, fetches the
        actual next-day close price and evaluates accuracy.

        Parameters
        ----------
        days_back : how many days back to check (default 10)

        Returns
        -------
        int: number of predictions updated
        """
        import yfinance as yf

        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        today  = datetime.now().strftime("%Y-%m-%d")

        sql = """
            SELECT id, ticker, date, direction, price_at_pred, timeframe
            FROM predictions
            WHERE correct IS NULL
              AND date >= ?
              AND date < ?
            ORDER BY ticker, date
        """
        with self._conn() as conn:
            pending = pd.read_sql(sql, conn, params=(cutoff, today))

        if len(pending) == 0:
            logger.info("No pending predictions to update")
            return 0

        logger.info(f"Updating outcomes for {len(pending)} pending predictions...")
        updated = 0

        # Group by ticker to minimise yfinance calls
        for ticker, group in pending.groupby("ticker"):
            try:
                # Fetch price data covering all pending dates
                min_date = group["date"].min()
                df = yf.download(
                    ticker,
                    start=min_date,
                    end=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index = pd.to_datetime(df.index).normalize()

                for _, row in group.iterrows():
                    pred_date = pd.to_datetime(row["date"])
                    # Find the next trading day's close
                    future = df[df.index > pred_date]
                    if len(future) == 0:
                        continue

                    actual_price = float(future["Close"].iloc[0])
                    pred_price   = float(row["price_at_pred"]) if row["price_at_pred"] else None

                    if pred_price and pred_price > 0:
                        actual_dir   = "UP" if actual_price > pred_price else "DOWN"
                        correct      = 1 if actual_dir == row["direction"] else 0
                    else:
                        actual_dir = None
                        correct    = None

                    with self._conn() as conn:
                        conn.execute(
                            """UPDATE predictions
                               SET actual_price=?, actual_direction=?, correct=?
                               WHERE id=?""",
                            (actual_price, actual_dir, correct, int(row["id"])),
                        )
                    updated += 1

            except Exception as e:
                logger.warning(f"Could not update outcomes for {ticker}: {e}")

        logger.success(f"Updated {updated} prediction outcomes")
        return updated

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_history(
        self,
        ticker:    Optional[str]  = None,
        timeframe: Optional[str]  = None,
        days_back: Optional[int]  = None,
        limit:     Optional[int]  = None,
    ) -> pd.DataFrame:
        """
        Retrieve prediction history as a DataFrame.

        Parameters
        ----------
        ticker    : filter by ticker (None = all)
        timeframe : filter by timeframe (None = all)
        days_back : only last N days (None = all)
        limit     : max rows to return

        Returns
        -------
        pd.DataFrame sorted by date descending
        """
        conditions = []
        params     = []

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)
        if days_back:
            cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            conditions.append("date >= ?")
            params.append(cutoff)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        lim   = f"LIMIT {limit}" if limit else ""

        sql = f"""
            SELECT * FROM predictions
            {where}
            ORDER BY date DESC, timestamp DESC
            {lim}
        """
        with self._conn() as conn:
            df = pd.read_sql(sql, conn, params=params)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df

    def get_accuracy_stats(
        self,
        ticker:    Optional[str] = None,
        timeframe: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> dict:
        """
        Compute accuracy statistics for evaluated predictions.

        Returns
        -------
        dict with:
            total, evaluated, correct, accuracy,
            up_accuracy, down_accuracy,
            win_streak, avg_confidence
        """
        df = self.get_history(ticker=ticker, timeframe=timeframe, days_back=days_back)
        evl = df[df["correct"].notna()].copy()

        if len(evl) == 0:
            return {
                "total": len(df), "evaluated": 0,
                "correct": 0, "accuracy": 0,
                "up_accuracy": 0, "down_accuracy": 0,
                "win_streak": 0, "avg_confidence": 0,
                "pending": len(df),
            }

        evl["correct"] = evl["correct"].astype(int)

        # Direction-specific accuracy
        up_mask   = evl["direction"] == "UP"
        dn_mask   = evl["direction"] == "DOWN"
        up_acc    = evl[up_mask]["correct"].mean()   if up_mask.any()   else 0
        dn_acc    = evl[dn_mask]["correct"].mean()   if dn_mask.any()   else 0

        # Current win streak
        streak = 0
        for c in evl.sort_values("date")["correct"].values[::-1]:
            if c == 1:
                streak += 1
            else:
                break

        return {
            "total":          len(df),
            "evaluated":      len(evl),
            "pending":        len(df) - len(evl),
            "correct":        int(evl["correct"].sum()),
            "accuracy":       round(float(evl["correct"].mean()), 4),
            "up_accuracy":    round(float(up_acc), 4),
            "down_accuracy":  round(float(dn_acc), 4),
            "win_streak":     streak,
            "avg_confidence": round(float(evl["confidence"].mean()), 4)
                              if "confidence" in evl.columns else 0,
        }

    def get_accuracy_by_ticker(self, days_back: int = 90) -> pd.DataFrame:
        """Accuracy stats grouped by ticker."""
        df  = self.get_history(days_back=days_back)
        evl = df[df["correct"].notna()].copy()
        if len(evl) == 0:
            return pd.DataFrame()
        evl["correct"] = evl["correct"].astype(int)
        return (
            evl.groupby("ticker")
               .agg(
                   predictions = ("correct", "count"),
                   correct     = ("correct", "sum"),
                   accuracy    = ("correct", "mean"),
                   avg_conf    = ("confidence", "mean"),
               )
               .round(4)
               .sort_values("accuracy", ascending=False)
               .reset_index()
        )

    def get_daily_accuracy(self, days_back: int = 90) -> pd.DataFrame:
        """Daily accuracy trend for charting."""
        df  = self.get_history(days_back=days_back)
        evl = df[df["correct"].notna()].copy()
        if len(evl) == 0:
            return pd.DataFrame()
        evl["correct"] = evl["correct"].astype(int)
        evl["date"]    = pd.to_datetime(evl["date"]).dt.date
        return (
            evl.groupby("date")
               .agg(
                   predictions = ("correct", "count"),
                   correct     = ("correct", "sum"),
                   accuracy    = ("correct", "mean"),
               )
               .round(4)
               .reset_index()
               .sort_values("date")
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def export_csv(self, path: Optional[str] = None) -> str:
        """Export all predictions to CSV. Returns file path."""
        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(project_path("results", f"predictions_{ts}.csv"))
        ensure_dirs(Path(path).parent)
        df = self.get_history()
        df.to_csv(path, index=False)
        logger.success(f"Exported {len(df)} predictions → {path}")
        return path

    def export_excel(self, path: Optional[str] = None) -> str:
        """Export predictions + accuracy stats to multi-sheet Excel."""
        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(project_path("results", f"predictions_{ts}.xlsx"))
        ensure_dirs(Path(path).parent)

        df_all      = self.get_history()
        df_by_tick  = self.get_accuracy_by_ticker()
        df_daily    = self.get_daily_accuracy()
        stats       = self.get_accuracy_stats()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="All Predictions", index=False)
            df_by_tick.to_excel(writer, sheet_name="Accuracy by Ticker", index=False)
            df_daily.to_excel(writer, sheet_name="Daily Accuracy", index=False)
            pd.DataFrame([stats]).to_excel(writer, sheet_name="Summary", index=False)

        logger.success(f"Exported Excel → {path}  ({len(df_all)} rows)")
        return path

    # ── Utility ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def delete_ticker(self, ticker: str) -> int:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE ticker=?", (ticker,))
            return cursor.rowcount

    def clear_all(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM predictions")
        logger.warning("All predictions deleted from store")