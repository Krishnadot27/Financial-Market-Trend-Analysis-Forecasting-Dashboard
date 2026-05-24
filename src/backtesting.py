"""
src/backtesting.py
───────────────────
Production-grade backtesting engine for NSE stock strategies.

Features:
  - Long-only daily rebalancing backtest
  - Transaction costs (NSE brokerage + STT + exchange charges)
  - Slippage modelling
  - Position sizing (fixed fractional)
  - Compares ML strategy vs Buy-and-Hold benchmark
  - Full performance report: Sharpe, Drawdown, Calmar, Alpha
  - Plotly interactive charts saved to results/

Usage:
    from src.backtesting import Backtester, run_backtest

    bt = Backtester(prices, signals, transaction_cost=0.001)
    results = bt.run()
    bt.report()
    bt.plot(ticker='RELIANCE.NS', save_path='results/backtest.html')
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation import ModelEvaluator
from src.utils import ensure_dirs, project_path


# ══════════════════════════════════════════════════════════════════════════════
# Backtester
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Simple long-only daily rebalancing backtest.

    Signal logic:
      signal > threshold  →  LONG  (hold 1 unit)
      signal <= threshold →  CASH  (exit to cash)

    Cost model (realistic NSE estimates):
      transaction_cost = 0.001  (0.1%: brokerage + STT + exchange)
      slippage         = 0.0005 (0.05%: market impact on entry/exit)

    Parameters
    ----------
    prices           : pd.Series  Actual closing prices (DatetimeIndex)
    signals          : pd.Series  Model predictions — float (return) or int (0/1)
    transaction_cost : float      Round-trip cost as fraction of trade value
    slippage         : float      Market impact cost per trade
    signal_threshold : float      Signal value above which we go LONG (default 0)
    """

    def __init__(
        self,
        prices:           pd.Series,
        signals:          pd.Series,
        transaction_cost: float = 0.001,
        slippage:         float = 0.0005,
        signal_threshold: float = 0.0,
    ) -> None:
        if len(prices) != len(signals):
            raise ValueError(
                f"prices ({len(prices)}) and signals ({len(signals)}) "
                "must have the same length."
            )
        self.prices           = prices.copy()
        self.signals          = signals.copy()
        self.transaction_cost = transaction_cost
        self.slippage         = slippage
        self.signal_threshold = signal_threshold
        self.results: Optional[pd.DataFrame] = None

    # ── Core backtest logic ───────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest and return a results DataFrame.

        Columns in results:
          price          : actual closing price
          signal         : model signal value
          position       : 1=Long, 0=Cash (based on today's signal)
          position_lag   : yesterday's position (what we actually hold today)
          trade          : 1 if position changed vs yesterday
          cost           : transaction cost + slippage on trade days
          price_return   : actual daily price return
          strat_return   : strategy return = position_lag * price_return - cost
          bh_return      : buy-and-hold daily return (benchmark)
          strat_cum      : cumulative strategy value (starts at 1.0)
          bh_cum         : cumulative buy-and-hold value (starts at 1.0)
        """
        df = pd.DataFrame({
            "price":  self.prices,
            "signal": self.signals,
        }).dropna()

        # ── Positions ─────────────────────────────────────────────────────────
        # Position based on TODAY's signal → executed at TOMORROW's open
        # This is the correct causal ordering (no lookahead)
        df["position"]   = (df["signal"] > self.signal_threshold).astype(int)
        df["position_lag"] = df["position"].shift(1).fillna(0).astype(int)

        # ── Trade detection & costs ───────────────────────────────────────────
        # A trade occurs whenever position changes (entry or exit)
        df["trade"] = df["position"].diff().abs().fillna(0)
        df["cost"]  = df["trade"] * (self.transaction_cost + self.slippage)

        # ── Returns ───────────────────────────────────────────────────────────
        df["price_return"] = df["price"].pct_change().fillna(0)

        # Strategy: hold yesterday's position, earn today's return, pay cost on trades
        df["strat_return"] = (
            df["position_lag"] * df["price_return"] - df["cost"]
        )
        # Benchmark: always hold
        df["bh_return"] = df["price_return"]

        # ── Cumulative returns ─────────────────────────────────────────────────
        df["strat_cum"] = (1 + df["strat_return"]).cumprod()
        df["bh_cum"]    = (1 + df["bh_return"]).cumprod()

        self.results = df
        logger.info(
            f"Backtest complete: {len(df)} trading days  |  "
            f"Trades: {int(df['trade'].sum())}  |  "
            f"Long days: {int(df['position_lag'].sum())} "
            f"({df['position_lag'].mean()*100:.1f}%)"
        )
        return df

    # ── Performance report ────────────────────────────────────────────────────

    def report(self) -> dict:
        """
        Print a full performance comparison between ML strategy and Buy-and-Hold.
        Returns a dict with all metrics for programmatic use.
        """
        if self.results is None:
            raise RuntimeError("Call .run() before .report()")

        df  = self.results
        ev  = ModelEvaluator()

        strat_m = ev.trading_report(df["strat_return"], label="ML Strategy")
        bh_m    = ev.trading_report(df["bh_return"],    label="Buy & Hold")

        # Alpha vs benchmark
        alpha = strat_m["Annual Return %"] - bh_m["Annual Return %"]

        print(f"\n{'═'*55}")
        print(f"  Strategy vs Buy-and-Hold Comparison")
        print(f"{'─'*55}")
        print(f"  {'Metric':<25} {'ML Strategy':>12} {'Buy&Hold':>12}")
        print(f"{'─'*55}")

        shared_keys = [k for k in strat_m if k in bh_m]
        for k in shared_keys:
            sv = strat_m[k]
            bv = bh_m[k]
            # Highlight rows where strategy wins
            better = ""
            if k in ["Total Return %", "Annual Return %", "Sharpe Ratio",
                      "Calmar Ratio", "Win Rate %"]:
                better = " ✔" if sv > bv else " ✘"
            elif k in ["Max Drawdown %"]:
                better = " ✔" if sv > bv else " ✘"  # less negative = better
            print(f"  {k:<25} {str(sv):>12} {str(bv):>12}{better}")

        print(f"{'─'*55}")
        print(f"  {'Alpha (vs B&H)':<25} {alpha:>+11.2f}%")
        print(f"  {'Total Trades':<25} {int(df['trade'].sum()):>12}")
        print(f"  {'Long %':<25} {df['position_lag'].mean()*100:>11.1f}%")
        print(f"{'═'*55}\n")

        return {"strategy": strat_m, "benchmark": bh_m, "alpha": alpha}

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot(
        self,
        ticker:    str = "NSE Stock",
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Generate an interactive Plotly backtest dashboard with 3 panels:
          1. Cumulative returns (Strategy vs Buy-and-Hold)
          2. Daily signal values
          3. Price chart with trade markers
        """
        if self.results is None:
            raise RuntimeError("Call .run() before .plot()")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly not installed — skipping plot. "
                           "Run: pip install plotly")
            return

        df = self.results

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Cumulative Returns",
                "Model Signal",
                f"{ticker} Price",
            ),
            row_heights=[0.50, 0.20, 0.30],
            vertical_spacing=0.06,
        )

        # Panel 1: Cumulative returns
        fig.add_trace(go.Scatter(
            x=df.index, y=df["strat_cum"],
            name="ML Strategy",
            line=dict(color="#2563EB", width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["bh_cum"],
            name="Buy & Hold",
            line=dict(color="#DC2626", width=1.5, dash="dash"),
        ), row=1, col=1)

        # Drawdown shading
        roll_max = df["strat_cum"].cummax()
        drawdown = (df["strat_cum"] - roll_max) / roll_max
        fig.add_trace(go.Scatter(
            x=df.index, y=drawdown,
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(220,38,38,0.15)",
            line=dict(color="rgba(220,38,38,0.3)", width=0.5),
        ), row=1, col=1)

        # Panel 2: Signal
        colors = ["#16A34A" if s > self.signal_threshold else "#DC2626"
                  for s in df["signal"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["signal"],
            name="Signal",
            marker_color=colors,
            showlegend=False,
        ), row=2, col=1)

        # Signal threshold line
        fig.add_hline(
            y=self.signal_threshold,
            line_dash="dot", line_color="gray",
            row=2, col=1,
        )

        # Panel 3: Price + trade markers
        fig.add_trace(go.Scatter(
            x=df.index, y=df["price"],
            name="Price",
            line=dict(color="#1E293B", width=1),
            showlegend=False,
        ), row=3, col=1)

        # Buy signals (green triangles up)
        buys = df[(df["trade"] == 1) & (df["position"] == 1)]
        if len(buys):
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["price"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", color="#16A34A", size=8),
            ), row=3, col=1)

        # Sell signals (red triangles down)
        sells = df[(df["trade"] == 1) & (df["position"] == 0)]
        if len(sells):
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["price"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", color="#DC2626", size=8),
            ), row=3, col=1)

        fig.update_layout(
            title=dict(
                text=f"Backtest: {ticker} — ML Strategy vs Buy & Hold",
                font=dict(size=16),
            ),
            height=750,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(title_text="Signal", row=2, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=3, col=1)

        if save_path:
            ensure_dirs(Path(save_path).parent)
            fig.write_html(str(save_path))
            logger.success(f"Backtest chart saved → {save_path}")
        else:
            fig.show()

    # ── Monthly returns heatmap ───────────────────────────────────────────────

    def monthly_returns_heatmap(
        self,
        save_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Compute monthly return table (rows=years, cols=months).
        Useful for spotting seasonal patterns and bad months.
        """
        if self.results is None:
            raise RuntimeError("Call .run() first")

        monthly = (
            (1 + self.results["strat_return"])
            .resample("ME")
            .prod() - 1
        ) * 100

        table = monthly.groupby([
            monthly.index.year,
            monthly.index.month,
        ]).first().unstack(level=1)

        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        table.columns = [month_names[m-1] for m in table.columns]

        # Annual return column
        table["Annual"] = (
            (1 + monthly / 100)
            .groupby(monthly.index.year)
            .prod() - 1
        ) * 100

        print("\nMonthly Returns (%) — ML Strategy")
        print(table.round(1).to_string())

        try:
            import plotly.graph_objects as go
            z     = table.values.tolist()
            years = [str(y) for y in table.index.tolist()]
            cols  = table.columns.tolist()

            fig = go.Figure(go.Heatmap(
                z=z, x=cols, y=years,
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z],
                texttemplate="%{text}",
                showscale=True,
            ))
            fig.update_layout(
                title="Monthly Returns Heatmap (%) — ML Strategy",
                template="plotly_white",
                height=max(300, len(years) * 40 + 100),
            )
            if save_path:
                fig.write_html(str(save_path))
                logger.success(f"Heatmap saved → {save_path}")
            else:
                fig.show()
        except ImportError:
            pass

        return table


# ══════════════════════════════════════════════════════════════════════════════
# Convenience runner
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df_features:      pd.DataFrame,
    model,
    target_col:       str   = "Target_Return",
    price_col:        str   = "Close",
    transaction_cost: float = 0.001,
    slippage:         float = 0.0005,
    ticker:           str   = "Stock",
    save_dir:         Optional[str | Path] = None,
) -> dict:
    """
    End-to-end backtest helper.

    Takes a feature DataFrame and a fitted model, generates predictions,
    runs the backtest, prints the report, and saves charts.

    Parameters
    ----------
    df_features      : Output of FeatureEngineer.build() — must include price col
    model            : Any fitted model with .predict(X) method
    target_col       : Target used to train the model
    price_col        : Column to use as actual price series
    transaction_cost : Brokerage + STT estimate
    slippage         : Market impact estimate
    ticker           : Display name for charts
    save_dir         : Directory to save HTML charts (None = show inline)

    Returns
    -------
    dict with 'results' DataFrame, 'metrics' dict, 'backtester' object
    """
    from src.feature_engineering import FeatureEngineer

    # ── Get features and prices ────────────────────────────────────────────────
    feature_cols = FeatureEngineer.get_feature_cols(df_features)
    X            = df_features[feature_cols]
    prices       = df_features[price_col]

    # ── Generate signals from model ────────────────────────────────────────────
    signals = pd.Series(model.predict(X), index=X.index, name="signal")

    logger.info(
        f"Running backtest for {ticker}  |  "
        f"Signal range: [{signals.min():.4f}, {signals.max():.4f}]  |  "
        f"Rows: {len(signals)}"
    )

    # ── Run backtest ───────────────────────────────────────────────────────────
    bt      = Backtester(prices, signals, transaction_cost, slippage)
    results = bt.run()
    metrics = bt.report()

    # ── Save charts ────────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        ensure_dirs(save_dir)
        safe_ticker = ticker.replace(".", "_").replace("^", "IDX_")
        bt.plot(
            ticker=ticker,
            save_path=save_dir / f"backtest_{safe_ticker}.html",
        )
        bt.monthly_returns_heatmap(
            save_path=save_dir / f"monthly_returns_{safe_ticker}.html",
        )

    return {
        "results":    results,
        "metrics":    metrics,
        "backtester": bt,
    }