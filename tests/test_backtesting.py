"""
tests/test_backtesting.py
──────────────────────────
Unit tests for the backtesting engine.
Run: pytest tests/test_backtesting.py -v
"""

import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stubs
for mod_name in ["loguru", "yfinance", "tqdm"]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if mod_name == "loguru":
            class _L:
                def info(s,*a,**k): pass
                def debug(s,*a,**k): pass
                def warning(s,*a,**k): pass
                def error(s,*a,**k): pass
                def success(s,*a,**k): pass
            m.logger = _L()
        if mod_name == "tqdm":
            m.tqdm = lambda x, **k: x
        sys.modules[mod_name] = m

import numpy as np
import pandas as pd
import pytest

from src.backtesting import Backtester


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_market():
    """300 days of zero-return market (price = constant)."""
    n     = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(1000.0, index=dates, name="price")


@pytest.fixture
def trending_market():
    """300 days of steadily rising market."""
    n     = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 1000 * (1.0005 ** np.arange(n))   # +0.05%/day
    return pd.Series(price, index=dates, name="price")


@pytest.fixture
def random_market():
    """300 days of random-walk market (realistic)."""
    np.random.seed(42)
    n     = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    ret   = np.random.randn(n) * 0.01
    price = 1000 * np.cumprod(1 + ret)
    return pd.Series(price, index=dates, name="price")


def always_long(prices):
    """Signal that is always above threshold → always invested."""
    return pd.Series(1.0, index=prices.index)


def always_cash(prices):
    """Signal that is always below threshold → always in cash."""
    return pd.Series(-1.0, index=prices.index)


def perfect_signal(prices):
    """Signal that correctly predicts direction every day."""
    returns = prices.pct_change().fillna(0)
    return returns  # positive return → buy, negative → sell


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_raises_on_length_mismatch(self, random_market):
        short_signal = pd.Series(1.0, index=random_market.index[:10])
        with pytest.raises(ValueError, match="same length"):
            Backtester(random_market, short_signal)

    def test_default_params(self, random_market):
        bt = Backtester(random_market, always_long(random_market))
        assert bt.transaction_cost == 0.001
        assert bt.slippage         == 0.0005
        assert bt.signal_threshold == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Run method
# ══════════════════════════════════════════════════════════════════════════════

class TestRun:

    def test_returns_dataframe(self, random_market):
        bt = Backtester(random_market, always_long(random_market))
        result = bt.run()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, random_market):
        bt = Backtester(random_market, always_long(random_market))
        result = bt.run()
        for col in ["position", "position_lag", "trade", "cost",
                    "price_return", "strat_return", "bh_return",
                    "strat_cum", "bh_cum"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_position_is_binary(self, random_market):
        signals = pd.Series(np.random.randn(len(random_market)),
                            index=random_market.index)
        bt     = Backtester(random_market, signals)
        result = bt.run()
        assert set(result["position"].unique()).issubset({0, 1})

    def test_always_long_matches_bh_minus_costs(self, trending_market):
        """
        Always-long strategy should match Buy-and-Hold minus transaction costs.
        The only difference is the cost on day 1 (entry) and day -1 (exit).
        """
        bt     = Backtester(trending_market, always_long(trending_market),
                            transaction_cost=0.001, slippage=0.0)
        result = bt.run()
        # Non-trade days: strat_return should equal bh_return exactly
        non_trade = result[result["trade"] == 0]
        if len(non_trade) > 0:
            diff = (non_trade["strat_return"] - non_trade["bh_return"]).abs()
            assert diff.max() < 1e-10, "Non-trade day returns differ from B&H"

    def test_always_cash_has_zero_returns(self, random_market):
        """
        Always-cash strategy: position=0 always → strat_return = -cost on trades.
        On non-trade days strat_return must be exactly 0.
        """
        bt     = Backtester(random_market, always_cash(random_market))
        result = bt.run()
        non_trade_returns = result.loc[result["trade"] == 0, "strat_return"]
        assert (non_trade_returns == 0).all(), \
            "Cash strategy should have zero returns on non-trade days"

    def test_costs_only_on_trade_days(self, random_market):
        bt     = Backtester(random_market, always_long(random_market))
        result = bt.run()
        no_trade_costs = result.loc[result["trade"] == 0, "cost"]
        assert (no_trade_costs == 0).all(), \
            "Costs should be 0 on days with no position change"

    def test_cumulative_starts_at_one(self, random_market):
        bt     = Backtester(random_market, always_long(random_market))
        result = bt.run()
        assert abs(result["strat_cum"].iloc[0] - (1 + result["strat_return"].iloc[0])) < 1e-9
        assert abs(result["bh_cum"].iloc[0] - (1 + result["bh_return"].iloc[0])) < 1e-9

    def test_no_lookahead_in_positions(self, random_market):
        """
        position_lag[t] must equal position[t-1].
        This ensures we trade on yesterday's signal, not today's.
        """
        bt     = Backtester(random_market, always_long(random_market))
        result = bt.run()
        for i in range(1, min(20, len(result))):
            today_lag  = result["position_lag"].iloc[i]
            yesterday  = result["position"].iloc[i-1]
            assert today_lag == yesterday, \
                f"LOOKAHEAD at row {i}: position_lag={today_lag} ≠ position[t-1]={yesterday}"

    def test_results_stored_after_run(self, random_market):
        bt = Backtester(random_market, always_long(random_market))
        assert bt.results is None
        bt.run()
        assert bt.results is not None


# ══════════════════════════════════════════════════════════════════════════════
# Performance logic
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance:

    def test_perfect_signal_beats_bh(self):
        """
        A perfect predictor avoids ALL down days and holds ALL up days.
        On a market with mixed returns, this always beats B&H.
        Signal is the next-day return shifted forward by 1 (true lookahead oracle).
        """
        np.random.seed(7)
        n      = 200
        dates  = pd.bdate_range("2020-01-01", periods=n)
        ret    = np.random.randn(n) * 0.01           # mixed daily returns
        prices = pd.Series(1000 * np.cumprod(1+ret), index=dates)
        # Oracle: signal at t = actual return at t (shift(-1) = lookahead oracle)
        # Positive signal → be long tomorrow → earn positive return
        oracle  = pd.Series(ret, index=dates)       # signal = actual next return
        bt      = Backtester(prices, oracle, transaction_cost=0.0, slippage=0.0)
        result  = bt.run()
        strat   = result["strat_cum"].iloc[-1]
        bh      = result["bh_cum"].iloc[-1]
        # Oracle avoids all down days so must be >= B&H
        assert strat >= bh, \
            f"Oracle ({strat:.3f}) should be >= B&H ({bh:.3f})"

    def test_transaction_costs_reduce_returns(self, random_market):
        """Higher transaction costs must reduce final strategy returns."""
        signals = pd.Series(np.random.randn(len(random_market)),
                            index=random_market.index)
        bt_low  = Backtester(random_market, signals, transaction_cost=0.0001)
        bt_high = Backtester(random_market, signals, transaction_cost=0.01)
        r_low   = bt_low.run()["strat_cum"].iloc[-1]
        r_high  = bt_high.run()["strat_cum"].iloc[-1]
        assert r_low >= r_high, \
            "Higher costs should reduce returns"

    def test_zero_cost_alwayslong_equals_bh(self, random_market):
        """
        Zero-cost always-long strategy must exactly equal Buy-and-Hold.
        """
        bt     = Backtester(random_market, always_long(random_market),
                            transaction_cost=0.0, slippage=0.0)
        result = bt.run()
        # After day 1 (position_lag kicks in), they should track perfectly
        aligned = result.iloc[2:]   # Skip first 2 days (position ramp-up)
        diff    = (aligned["strat_cum"] - aligned["bh_cum"]).abs()
        assert diff.max() < 1e-8, \
            f"Zero-cost always-long diverged from B&H: max diff = {diff.max()}"

    def test_report_raises_before_run(self, random_market):
        bt = Backtester(random_market, always_long(random_market))
        with pytest.raises(RuntimeError, match="run()"):
            bt.report()

    def test_report_returns_dict(self, random_market):
        bt     = Backtester(random_market, always_long(random_market))
        bt.run()
        result = bt.report()
        assert isinstance(result, dict)
        assert "strategy" in result
        assert "benchmark" in result
        assert "alpha" in result

    def test_alpha_calculation(self, random_market):
        """Alpha = strategy annual return - benchmark annual return."""
        signals = always_long(random_market)
        bt      = Backtester(random_market, signals)
        bt.run()
        metrics = bt.report()
        alpha   = metrics["alpha"]
        expected = (
            metrics["strategy"]["Annual Return %"] -
            metrics["benchmark"]["Annual Return %"]
        )
        assert abs(alpha - expected) < 1e-6

    def test_monthly_returns_shape(self, random_market):
        bt     = Backtester(random_market, always_long(random_market))
        bt.run()
        table  = bt.monthly_returns_heatmap()
        # Columns should be month names + Annual
        assert "Annual" in table.columns
        assert len(table.columns) <= 13   # max 12 months + Annual


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_all_positive_signals_all_long(self, random_market):
        """All signals > 0 → position = 1 every day."""
        bt     = Backtester(random_market, always_long(random_market))
        result = bt.run()
        # After day 1, position_lag should always be 1
        assert result["position"].eq(1).all()

    def test_all_negative_signals_all_cash(self, random_market):
        """All signals < 0 → position = 0 every day."""
        bt     = Backtester(random_market, always_cash(random_market))
        result = bt.run()
        assert result["position"].eq(0).all()

    def test_custom_threshold(self, random_market):
        """Custom threshold=0.5 → only enter when signal > 0.5."""
        signals = pd.Series(
            np.where(np.arange(len(random_market)) % 2 == 0, 0.8, 0.2),
            index=random_market.index,
        )
        bt     = Backtester(random_market, signals, signal_threshold=0.5)
        result = bt.run()
        # Even-indexed days should be long, odd-indexed cash
        assert result["position"].iloc[0] == 1   # signal=0.8 > 0.5
        assert result["position"].iloc[1] == 0   # signal=0.2 < 0.5

    def test_short_series(self):
        """Backtest should work on small series (≥2 rows)."""
        n      = 10
        dates  = pd.bdate_range("2020-01-01", periods=n)
        prices = pd.Series(1000.0 * np.cumprod(1 + np.random.randn(n)*0.01),
                           index=dates)
        sigs   = pd.Series(1.0, index=dates)
        bt     = Backtester(prices, sigs)
        result = bt.run()
        assert len(result) == n