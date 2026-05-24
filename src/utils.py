"""
src/utils.py
────────────
Shared utilities: config loading, logging setup, reproducibility,
directory helpers, and timing decorators.
"""

from __future__ import annotations

import os
import random
import time
import functools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger


# ── Project root (two levels up from this file: src/ → project/) ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load config.yaml from the project root (or a custom path).

    Usage:
        cfg = load_config()
        tickers = cfg['data']['tickers']
    """
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

_logger_configured = False

def setup_logger(log_dir: str | Path | None = None, level: str = "INFO") -> None:
    """
    Configure loguru logger — writes to console + rotating log file.

    Call once at the top of any script/notebook:
        from src.utils import setup_logger
        setup_logger()
    """
    global _logger_configured
    if _logger_configured:
        return

    logger.remove()  # Remove default handler

    # Console: coloured, readable
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    # File: full detail, rotating at 10 MB
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=str(log_path / "stock_ml_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        )

    _logger_configured = True


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Directory helpers
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (and parents) if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    """Return an absolute path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


# ══════════════════════════════════════════════════════════════════════════════
# Timing decorator
# ══════════════════════════════════════════════════════════════════════════════

def timer(func):
    """Decorator — logs how long a function takes."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# DataFrame helpers
# ══════════════════════════════════════════════════════════════════════════════

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns produced by yfinance multi-ticker downloads.

    Example: ('Close', 'RELIANCE.NS') → 'Close_RELIANCE.NS'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns]
    return df


def safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Division that replaces inf/nan with `fill` value."""
    result = a / b.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def pct_change_clipped(series: pd.Series, clip: float = 0.5) -> pd.Series:
    """
    Percentage change, clipped at ±clip to remove data errors
    (e.g. stock splits not adjusted for, erroneous price spikes).
    """
    chg = series.pct_change()
    return chg.clip(-clip, clip)


# ══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ══════════════════════════════════════════════════════════════════════════════

def check_no_lookahead(df: pd.DataFrame, target_col: str) -> None:
    """
    Sanity check: ensure the target column is not in the feature set
    that would be used to train. Raises AssertionError if feature
    correlates perfectly with same-day target (indicates leakage).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        if col in df.columns and df[target_col].corr(df[col]) > 0.999:
            raise ValueError(
                f"Potential lookahead leakage: '{col}' is nearly perfectly "
                f"correlated with target '{target_col}'. "
                "Check that all features are lagged by at least 1 period."
            )
    logger.info(f"Lookahead check passed for target='{target_col}'")


def summarise_df(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a concise quality summary of a DataFrame."""
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Shape       : {df.shape}")
    print(f"  Date range  : {df.index[0]} → {df.index[-1]}")
    print(f"  Null values : {df.isnull().sum().sum()}")
    print(f"  Duplicates  : {df.index.duplicated().sum()}")
    print(f"  dtypes      : {df.dtypes.value_counts().to_dict()}")
    print(f"{'─'*55}\n")