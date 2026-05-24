"""
tests/test_portfolio.py
────────────────────────
Tests for multi-ticker Portfolio engine.
Run: pytest tests/test_portfolio.py -v
"""

import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for mod_name in ["loguru","yfinance","tqdm"]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if mod_name=="loguru":
            class _L:
                def info(s,*a,**k): pass
                def debug(s,*a,**k): pass
                def warning(s,*a,**k): pass
                def error(s,*a,**k): pass
                def success(s,*a,**k): pass
            m.logger=_L()
        if mod_name=="tqdm": m.tqdm=lambda x,**k:x
        sys.modules[mod_name]=m

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def market_data():
    """3 tickers × 300 days of synthetic OHLCV."""
    np.random.seed(42)
    n       = 300
    dates   = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["AAAA.NS", "BBBB.NS", "CCCC.NS"]

    prices  = {}
    signals = {}
    for i, t in enumerate(tickers):
        ret    = np.random.randn(n) * 0.01 + 0.0002 * (i + 1)
        price  = 1000 * np.cumprod(1 + ret)
        prices[t]  = pd.Series(price, index=dates)
        signals[t] = pd.Series(ret + np.random.randn(n)*0.005, index=dates)

    return tickers, prices, signals


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioInit:

    def test_invalid_strategy_raises(self, market_data):
        tickers, prices, signals = market_data
        with pytest.raises(ValueError, match="strategy"):
            Portfolio(signals, prices, strategy="bad_strategy")

    def test_valid_strategies_accepted(self, market_data):
        tickers, prices, signals = market_data
        for strat in ["equal_weight","momentum_weight","signal_weight","min_variance"]:
            pf = Portfolio(signals, prices, strategy=strat)
            assert pf.strategy == strat

    def test_tickers_detected(self, market_data):
        tickers, prices, signals = market_data
        pf = Portfolio(signals, prices)
        assert set(pf.tickers) == set(tickers)


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioRun:

    def test_run_returns_dataframe(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices, strategy="equal_weight")
        r  = pf.run()
        assert isinstance(r, pd.DataFrame)

    def test_required_columns(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices)
        r  = pf.run()
        for col in ["portfolio_return","net_return","cum_return","bh_cum","cost"]:
            assert col in r.columns

    def test_weight_columns_present(self, market_data):
        tickers, prices, signals = market_data
        pf = Portfolio(signals, prices)
        r  = pf.run()
        for t in tickers:
            assert f"{t}_weight" in r.columns

    def test_cum_return_starts_positive(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices)
        r  = pf.run()
        assert r["cum_return"].iloc[0] > 0

    def test_costs_non_negative(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices)
        r  = pf.run()
        assert (r["cost"] >= 0).all()

    def test_zero_cost_net_equals_gross(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices, transaction_cost=0.0)
        r  = pf.run()
        diff = (r["net_return"] - r["portfolio_return"]).abs()
        assert diff.max() < 1e-10

    def test_results_stored(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices)
        assert pf.results is None
        pf.run()
        assert pf.results is not None


# ══════════════════════════════════════════════════════════════════════════════
# Strategies
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategies:

    def _run(self, signals, prices, strategy):
        pf = Portfolio(signals, prices, strategy=strategy, transaction_cost=0.0)
        return pf.run(), pf

    def test_equal_weight_weights_sum_to_le_one(self, market_data):
        tickers, prices, signals = market_data
        r, pf = self._run(signals, prices, "equal_weight")
        w_cols = [f"{t}_weight" for t in tickers]
        row_sums = r[w_cols].sum(axis=1)
        # Weights sum to ≤ 1 (can be 0 if all signals are negative)
        assert (row_sums <= 1.0 + 1e-9).all()

    def test_signal_weight_proportional(self, market_data):
        _, prices, _ = market_data
        dates = prices[list(prices.keys())[0]].index
        # All-positive signals: higher signal → higher weight
        signals_fixed = {
            "AAAA.NS": pd.Series(0.01, index=dates),  # low signal
            "BBBB.NS": pd.Series(0.05, index=dates),  # high signal
            "CCCC.NS": pd.Series(0.02, index=dates),  # medium signal
        }
        pf = Portfolio(signals_fixed, prices, strategy="signal_weight",
                       transaction_cost=0.0)
        pf.run()
        # BBBB should have highest weight
        w = pf.weights.mean()
        assert w["BBBB.NS"] > w["AAAA.NS"], \
            "Higher signal → higher weight in signal_weight strategy"

    def test_all_strategies_run_without_error(self, market_data):
        _, prices, signals = market_data
        for strat in ["equal_weight","momentum_weight","signal_weight","min_variance"]:
            pf = Portfolio(signals, prices, strategy=strat)
            r  = pf.run()
            assert len(r) > 0, f"Empty results for {strat}"

    def test_higher_cost_lower_net_return(self, market_data):
        _, prices, signals = market_data
        pf_lo = Portfolio(signals, prices, transaction_cost=0.0001)
        pf_hi = Portfolio(signals, prices, transaction_cost=0.01)
        r_lo  = pf_lo.run()["cum_return"].iloc[-1]
        r_hi  = pf_hi.run()["cum_return"].iloc[-1]
        assert r_lo >= r_hi, "Higher transaction cost should reduce returns"


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioReport:

    def test_report_raises_before_run(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices)
        with pytest.raises(RuntimeError, match="run()"):
            pf.report()

    def test_report_returns_dict(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices)
        pf.run()
        m = pf.report()
        assert isinstance(m, dict)
        assert "portfolio" in m
        assert "benchmark" in m
        assert "alpha" in m

    def test_alpha_calculated_correctly(self, market_data):
        _, prices, signals = market_data
        pf = Portfolio(signals, prices)
        pf.run()
        m = pf.report()
        expected = (
            m["portfolio"]["Annual Return %"] -
            m["benchmark"]["Annual Return %"]
        )
        assert abs(m["alpha"] - expected) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# Correlation matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationMatrix:

    def test_returns_dataframe(self, market_data):
        _, prices, signals = market_data
        pf   = Portfolio(signals, prices)
        corr = pf.correlation_matrix()
        assert isinstance(corr, pd.DataFrame)

    def test_diagonal_is_one(self, market_data):
        _, prices, signals = market_data
        pf   = Portfolio(signals, prices)
        corr = pf.correlation_matrix()
        diag = np.diag(corr.values)
        assert np.allclose(diag, 1.0)

    def test_symmetric(self, market_data):
        _, prices, signals = market_data
        pf   = Portfolio(signals, prices)
        corr = pf.correlation_matrix()
        assert np.allclose(corr.values, corr.values.T)