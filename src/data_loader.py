"""
src/data_loader.py
──────────────────
Production-grade NSE data ingestion.

Features:
  - Downloads OHLCV via yfinance for any NSE ticker (append .NS)
  - Local CSV caching — re-downloads only if file is stale or missing
  - Rigorous data quality validation (OHLC integrity, gaps, volume)
  - Returns a clean, timezone-naive DatetimeIndex DataFrame
  - Multi-ticker batch download with progress bar
  - Automatic handling of yfinance MultiIndex columns

Usage:
    from src.data_loader import NSEDataLoader, download_all_tickers

    # Single ticker
    loader = NSEDataLoader("RELIANCE.NS", "2014-01-01", "2024-12-31")
    df = loader.load()          # loads from cache if available

    # Batch download
    dfs = download_all_tickers(cfg['data']['tickers'], "2014-01-01", "2024-12-31")
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from src.utils import ensure_dirs, project_path, summarise_df, timer


# ── Constants ──────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CACHE_STALE_DAYS = 1          # Re-download if cached file is older than this


# ══════════════════════════════════════════════════════════════════════════════
# NSEDataLoader
# ══════════════════════════════════════════════════════════════════════════════

class NSEDataLoader:
    """
    Downloads and validates OHLCV data for a single NSE ticker.

    Parameters
    ----------
    ticker  : str   e.g. 'RELIANCE.NS'
    start   : str   'YYYY-MM-DD'
    end     : str   'YYYY-MM-DD'
    interval: str   '1d' | '1h' | '5m'  (default: '1d')
    cache_dir: Path  Directory to store CSVs (default: data/raw/)
    """

    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.ticker    = ticker.upper()
        self.start     = start
        self.end       = end
        self.interval  = interval
        self.cache_dir = Path(cache_dir) if cache_dir else project_path("data", "raw")
        ensure_dirs(self.cache_dir)

    # ── Cache path ─────────────────────────────────────────────────────────────
    @property
    def cache_path(self) -> Path:
        safe_ticker = self.ticker.replace(".", "_").replace("^", "IDX_")
        filename = f"{safe_ticker}_{self.start}_{self.end}_{self.interval}.csv"
        return self.cache_dir / filename

    def _is_cache_fresh(self) -> bool:
        """Return True if the cache file exists and was modified within CACHE_STALE_DAYS."""
        if not self.cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(self.cache_path.stat().st_mtime)
        return (datetime.now() - mtime) < timedelta(days=CACHE_STALE_DAYS)

    # ── Download ───────────────────────────────────────────────────────────────
    @timer
    def _download_from_yahoo(self) -> pd.DataFrame:
        """Fetch raw OHLCV from Yahoo Finance."""
        logger.info(f"Downloading {self.ticker}  [{self.start} → {self.end}]  interval={self.interval}")
        raw = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,     # Adjusts for splits & dividends automatically
            progress=False,
            threads=True,
        )
        if raw.empty:
            raise ValueError(
                f"yfinance returned empty DataFrame for '{self.ticker}'. "
                "Check the ticker symbol and date range."
            )
        return raw

    # ── Column normalisation ───────────────────────────────────────────────────
    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        yfinance sometimes returns MultiIndex columns when downloading
        a single ticker inside a batch call.  Flatten them.
        Also standardise column names to Title Case.
        """
        if isinstance(df.columns, pd.MultiIndex):
            # ('Close', 'RELIANCE.NS') → 'Close'
            df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]

        # Standardise: 'open' → 'Open', 'adj close' → 'Adj Close', etc.
        rename = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
            "adj close": "Adj Close", "adj_close": "Adj Close",
        }
        df.columns = [rename.get(c.lower(), c) for c in df.columns]
        return df

    # ── Validation ─────────────────────────────────────────────────────────────
    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality rules.  Raises ValueError on hard failures."""
        # 1. Required columns present
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {self.ticker}")

        # 2. Drop rows where ALL OHLCV are NaN (yfinance sometimes pads weekends)
        df = df.dropna(subset=REQUIRED_COLUMNS, how="all")

        # 3. Forward-fill up to 2 consecutive missing values (e.g. trading holidays)
        df = df.ffill(limit=2)

        # 4. Drop any remaining NaN rows
        before = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"{self.ticker}: dropped {dropped} NaN rows after ffill")

        # 5. OHLC integrity: High >= Close >= Low, High >= Open >= Low
        bad_high = (df["High"] < df["Close"]).sum()
        bad_low  = (df["Low"]  > df["Close"]).sum()
        if bad_high > 0:
            logger.warning(f"{self.ticker}: {bad_high} rows where High < Close — clipping")
            df["High"] = df[["High", "Close", "Open"]].max(axis=1)
        if bad_low > 0:
            logger.warning(f"{self.ticker}: {bad_low} rows where Low > Close — clipping")
            df["Low"] = df[["Low", "Close", "Open"]].min(axis=1)

        # 6. Volume must be positive
        zero_vol = (df["Volume"] <= 0).sum()
        if zero_vol > 0:
            logger.warning(f"{self.ticker}: {zero_vol} zero-volume rows — replacing with NaN then ffill")
            df.loc[df["Volume"] <= 0, "Volume"] = np.nan
            df["Volume"] = df["Volume"].ffill()

        # 7. No extreme single-day price moves (>50% → likely a split not caught)
        daily_chg = df["Close"].pct_change().abs()
        suspicious = (daily_chg > 0.5).sum()
        if suspicious > 0:
            logger.warning(
                f"{self.ticker}: {suspicious} daily moves >50% detected. "
                "These may be unadjusted splits — verify manually."
            )

        # 8. Minimum row requirement
        if len(df) < 200:
            raise ValueError(
                f"{self.ticker}: only {len(df)} rows — need at least 200 for meaningful ML."
            )

        return df

    # ── Index cleanup ──────────────────────────────────────────────────────────
    @staticmethod
    def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a clean, timezone-naive DatetimeIndex sorted ascending."""
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)  # remove tz info
        df = df.sort_index()
        df.index.name = "Date"
        return df

    # ── Public API ─────────────────────────────────────────────────────────────
    def load(self, force_download: bool = False) -> pd.DataFrame:
        """
        Main entry point.

        1. If cache is fresh, load from CSV.
        2. Otherwise download from Yahoo Finance, validate, and cache.

        Returns a clean OHLCV DataFrame indexed by Date.
        """
        if not force_download and self._is_cache_fresh():
            logger.info(f"Loading {self.ticker} from cache: {self.cache_path.name}")
            df = pd.read_csv(self.cache_path, index_col="Date", parse_dates=True)
        else:
            raw = self._download_from_yahoo()
            df  = self._normalise_columns(raw)
            df  = self._clean_index(df)
            df  = self._validate(df)
            df.to_csv(self.cache_path)
            logger.success(f"Cached {self.ticker} → {self.cache_path.name}  ({len(df)} rows)")

        # Always clean index on load (handles re-reads from CSV)
        df = self._clean_index(df)
        return df

    def load_with_summary(self, force_download: bool = False) -> pd.DataFrame:
        """load() + print a quality summary."""
        df = self.load(force_download)
        summarise_df(df, name=self.ticker)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Multi-ticker batch download
# ══════════════════════════════════════════════════════════════════════════════

def download_all_tickers(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Download and cache OHLCV data for a list of NSE tickers.

    Returns
    -------
    dict[ticker, DataFrame]  — only successfully loaded tickers are included.

    Example
    -------
    from src.utils import load_config
    cfg = load_config()
    dfs = download_all_tickers(
        cfg['data']['tickers'],
        cfg['data']['start_date'],
        cfg['data']['end_date'],
    )
    df_reliance = dfs['RELIANCE.NS']
    """
    results: dict[str, pd.DataFrame] = {}
    failed:  list[str] = []

    logger.info(f"Batch downloading {len(tickers)} tickers  [{start} → {end}]")

    for ticker in tqdm(tickers, desc="Downloading", unit="ticker"):
        try:
            loader = NSEDataLoader(ticker, start, end, interval, cache_dir)
            df = loader.load()
            results[ticker] = df
        except Exception as exc:
            logger.error(f"Failed to load {ticker}: {exc}")
            failed.append(ticker)

    logger.info(
        f"Batch complete: {len(results)} succeeded, {len(failed)} failed"
        + (f" — failed: {failed}" if failed else "")
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Train / Val / Test split
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
    val_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware train/val/test split.

    NEVER uses random shuffling.
    Timeline: |──────── TRAIN ────────|── VAL ──|── TEST ──|

    Parameters
    ----------
    df          : DatetimeIndex DataFrame (sorted ascending)
    test_ratio  : fraction of data held out for final evaluation
    val_ratio   : fraction held out for hyperparameter tuning

    Returns
    -------
    (train, val, test)  — three non-overlapping DataFrames
    """
    n = len(df)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_test - n_val

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train : n_train + n_val].copy()
    test  = df.iloc[n_train + n_val :].copy()

    logger.info(
        f"Split → Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} rows) | "
        f"Val: {val.index[0].date()} to {val.index[-1].date()} ({len(val)} rows) | "
        f"Test: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} rows)"
    )
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: load a single ticker with config
# ══════════════════════════════════════════════════════════════════════════════

def load_ticker_from_config(ticker: str, cfg: dict) -> pd.DataFrame:
    """
    Shortcut used in notebooks and training scripts.

    Example
    -------
    cfg = load_config()
    df  = load_ticker_from_config('RELIANCE.NS', cfg)
    """
    return NSEDataLoader(
        ticker,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        interval=cfg["data"]["interval"],
    ).load()