"""
src/portfolio.py
─────────────────
Multi-ticker portfolio construction and management.

Strategies implemented:
  1. EqualWeight      — 1/N allocation across all tickers
  2. MomentumWeight   — overweight recent winners, underweight losers
  3. SignalWeight      — weight by model confidence (predicted return magnitude)
  4. MinVariance       — minimum variance optimisation (no expected return input)

Usage:
    from src.portfolio import Portfolio

    pf = Portfolio(signals_dict, prices_dict, strategy='signal_weight')
    results = pf.run()
    pf.report()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation import ModelEvaluator
from src.utils import ensure_dirs, project_path


class Portfolio:
    """
    Multi-ticker portfolio backtester.

    Parameters
    ----------
    signals_dict  : {ticker: pd.Series}  — model signal per ticker
    prices_dict   : {ticker: pd.Series}  — daily close price per ticker
    strategy      : allocation strategy
    rebalance_freq: 'D' (daily), 'W' (weekly), 'ME' (monthly)
    transaction_cost: cost per trade (applied to weight changes)
    """

    STRATEGIES = ("equal_weight", "momentum_weight", "signal_weight", "min_variance")

    def __init__(
        self,
        signals_dict:     dict[str, pd.Series],
        prices_dict:      dict[str, pd.Series],
        strategy:         str   = "signal_weight",
        rebalance_freq:   str   = "W",
        transaction_cost: float = 0.001,
        signal_threshold: float = 0.0,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.signals_dict     = signals_dict
        self.prices_dict      = prices_dict
        self.strategy         = strategy
        self.rebalance_freq   = rebalance_freq
        self.transaction_cost = transaction_cost
        self.signal_threshold = signal_threshold
        self.tickers          = list(signals_dict.keys())
        self.results: Optional[pd.DataFrame] = None

        # Build aligned DataFrames
        self.signals_df = pd.DataFrame(signals_dict).sort_index().ffill()
        self.prices_df  = pd.DataFrame(prices_dict).sort_index().ffill()
        self.returns_df = self.prices_df.pct_change().fillna(0)

        # Align index
        common_idx      = self.signals_df.index.intersection(self.prices_df.index)
        self.signals_df = self.signals_df.loc[common_idx]
        self.prices_df  = self.prices_df.loc[common_idx]
        self.returns_df = self.returns_df.loc[common_idx]

        logger.info(
            f"Portfolio: {len(self.tickers)} tickers  |  "
            f"Strategy: {strategy}  |  "
            f"Rows: {len(common_idx)}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Weight computation
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_weights(self) -> pd.DataFrame:
        """
        Compute daily portfolio weights for each ticker.
        Returns DataFrame (dates × tickers) summing to 1.0 each row.
        """
        n = len(self.tickers)

        if self.strategy == "equal_weight":
            # Pure 1/N — only invest in tickers with positive signal
            long_mask = (self.signals_df > self.signal_threshold).astype(float)
            row_sums  = long_mask.sum(axis=1).replace(0, np.nan)
            weights   = long_mask.div(row_sums, axis=0).fillna(0)

        elif self.strategy == "momentum_weight":
            # Weight by 20-day momentum (normalised)
            momentum = self.prices_df.pct_change(20).fillna(0)
            # Only long tickers with positive signal AND positive momentum
            long_mask = (
                (self.signals_df > self.signal_threshold) &
                (momentum > 0)
            ).astype(float)
            # Weight proportional to momentum strength
            pos_mom   = momentum.clip(lower=0) * long_mask
            row_sums  = pos_mom.sum(axis=1).replace(0, np.nan)
            weights   = pos_mom.div(row_sums, axis=0).fillna(0)
            # Fall back to equal weight on zero-sum rows
            zero_rows = weights.sum(axis=1) == 0
            if zero_rows.any():
                ew = long_mask.div(
                    long_mask.sum(axis=1).replace(0, np.nan), axis=0
                ).fillna(0)
                weights[zero_rows] = ew[zero_rows]

        elif self.strategy == "signal_weight":
            # Weight proportional to |signal| — higher confidence → more weight
            long_mask      = (self.signals_df > self.signal_threshold).astype(float)
            signal_abs     = self.signals_df.abs() * long_mask
            row_sums       = signal_abs.sum(axis=1).replace(0, np.nan)
            weights        = signal_abs.div(row_sums, axis=0).fillna(0)
            # Fallback: equal weight when all signals equal
            zero_rows = weights.sum(axis=1) == 0
            if zero_rows.any():
                ew = long_mask.div(
                    long_mask.sum(axis=1).replace(0, np.nan), axis=0
                ).fillna(0)
                weights[zero_rows] = ew[zero_rows]

        elif self.strategy == "min_variance":
            # Minimum variance using rolling 60-day covariance
            weights = pd.DataFrame(index=self.returns_df.index,
                                   columns=self.tickers, dtype=float)
            window = 60
            for i in range(len(self.returns_df)):
                if i < window:
                    # Equal weight until we have enough history
                    weights.iloc[i] = 1.0 / n
                    continue
                ret_window = self.returns_df.iloc[i-window:i]
                try:
                    cov    = ret_window.cov().values
                    inv    = np.linalg.pinv(cov)
                    ones   = np.ones(n)
                    raw    = inv @ ones
                    raw    = np.maximum(raw, 0)  # long-only constraint
                    total  = raw.sum()
                    w      = raw / total if total > 0 else ones / n
                    weights.iloc[i] = w
                except Exception:
                    weights.iloc[i] = 1.0 / n
            weights = weights.astype(float)

        return weights

    # ══════════════════════════════════════════════════════════════════════════
    # Rebalancing
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_rebalancing(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rebalancing frequency — only update weights at rebalance dates,
        hold previous weights in between. This reduces transaction costs.
        """
        if self.rebalance_freq == "D":
            return weights

        # Get rebalance dates
        rebal_dates = weights.resample(self.rebalance_freq).last().index
        rebal_mask  = weights.index.isin(rebal_dates)

        rebalnced = weights.copy()
        last_w    = weights.iloc[0]

        for i, (date, row) in enumerate(weights.iterrows()):
            if rebal_mask[i]:
                last_w = row
            rebalnced.loc[date] = last_w

        return rebalnced

    # ══════════════════════════════════════════════════════════════════════════
    # Run backtest
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> pd.DataFrame:
        """
        Execute portfolio backtest.

        Returns DataFrame with columns:
          - {ticker}_weight   : portfolio weight each day
          - portfolio_return  : daily portfolio return
          - cost              : transaction cost on rebalance days
          - net_return        : portfolio_return - cost
          - cum_return        : cumulative portfolio value
          - bh_return         : equal-weight buy-and-hold benchmark
          - bh_cum            : cumulative B&H value
        """
        # Compute target weights
        raw_weights = self._compute_weights()
        weights     = self._apply_rebalancing(raw_weights)

        # Lag weights by 1 day (trade tomorrow at today's signal)
        weights_lag = weights.shift(1).fillna(0)

        # Portfolio daily return
        port_return = (weights_lag * self.returns_df).sum(axis=1)

        # Transaction costs — proportional to weight turnover
        turnover    = weights.diff().abs().sum(axis=1).fillna(0)
        cost        = turnover * self.transaction_cost
        net_return  = port_return - cost

        # Buy-and-hold benchmark (equal weight, always invested)
        bh_return = self.returns_df.mean(axis=1)

        df = pd.DataFrame({
            "portfolio_return": port_return,
            "cost":             cost,
            "net_return":       net_return,
            "bh_return":        bh_return,
            "cum_return":       (1 + net_return).cumprod(),
            "bh_cum":           (1 + bh_return).cumprod(),
        })

        # Add individual weights
        for t in self.tickers:
            df[f"{t}_weight"] = weights_lag[t]

        self.results  = df
        self.weights  = weights

        logger.info(
            f"Portfolio backtest complete  |  "
            f"Total trades: {(turnover > 0).sum()}  |  "
            f"Avg daily turnover: {turnover.mean()*100:.2f}%"
        )
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════════

    def report(self) -> dict:
        """Print full portfolio performance report."""
        if self.results is None:
            raise RuntimeError("Call .run() first")

        df = self.results
        ev = ModelEvaluator()

        port_m = ev.trading_report(df["net_return"],  label=f"Portfolio ({self.strategy})")
        bh_m   = ev.trading_report(df["bh_return"],   label="Equal-Weight B&H")

        alpha  = port_m["Annual Return %"] - bh_m["Annual Return %"]
        n_tickers_avg = (
            (self.weights > 0.01)
            .sum(axis=1)
            .mean()
        )

        print(f"\n{'═'*60}")
        print(f"  Portfolio Summary — {self.strategy}")
        print(f"{'─'*60}")
        print(f"  {'Metric':<28} {'Portfolio':>14} {'B&H':>10}")
        print(f"{'─'*60}")
        shared = [k for k in port_m if k in bh_m]
        for k in shared:
            pv = port_m[k]; bv = bh_m[k]
            better = ""
            if k in ["Total Return %","Annual Return %","Sharpe Ratio","Calmar Ratio","Win Rate %"]:
                better = " ✔" if pv > bv else " ✘"
            elif k == "Max Drawdown %":
                better = " ✔" if pv > bv else " ✘"
            print(f"  {k:<28} {str(pv):>14} {str(bv):>10}{better}")
        print(f"{'─'*60}")
        print(f"  {'Alpha':<28} {alpha:>+13.2f}%")
        print(f"  {'Avg tickers held':<28} {n_tickers_avg:>14.1f}")
        print(f"  {'Tickers':<28} {len(self.tickers):>14}")
        print(f"{'═'*60}\n")

        return {"portfolio": port_m, "benchmark": bh_m, "alpha": alpha}

    # ══════════════════════════════════════════════════════════════════════════
    # Visualisation
    # ══════════════════════════════════════════════════════════════════════════

    def plot(
        self,
        save_path: Optional[str] = None,
    ) -> None:
        """Interactive 3-panel portfolio dashboard."""
        if self.results is None:
            raise RuntimeError("Call .run() first")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly not installed — skipping plot")
            return

        df = self.results

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Portfolio vs Equal-Weight B&H",
                "Daily Net Return",
                "Portfolio Weights Over Time",
            ),
            row_heights=[0.45, 0.25, 0.30],
            vertical_spacing=0.06,
        )

        # Panel 1: Cumulative returns
        fig.add_trace(go.Scatter(
            x=df.index, y=df["cum_return"],
            name=f"Portfolio ({self.strategy})",
            line=dict(color="#3B82F6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bh_cum"],
            name="Equal-Weight B&H",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
        ), row=1, col=1)

        # Drawdown
        roll_max = df["cum_return"].cummax()
        dd = (df["cum_return"] - roll_max) / roll_max
        fig.add_trace(go.Scatter(
            x=df.index, y=dd,
            name="Drawdown", fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="rgba(239,68,68,0.4)", width=0.5),
            showlegend=False,
        ), row=1, col=1)

        # Panel 2: Daily returns
        colors = ["#22C55E" if r > 0 else "#EF4444" for r in df["net_return"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["net_return"],
            marker_color=colors, name="Daily Return", showlegend=False,
        ), row=2, col=1)

        # Panel 3: Stacked weight chart
        palette = ["#3B82F6","#22C55E","#F59E0B","#EF4444","#8B5CF6",
                   "#EC4899","#14B8A6","#F97316","#6366F1","#84CC16"]
        weight_cols = [c for c in df.columns if c.endswith("_weight")]
        for i, col in enumerate(weight_cols):
            ticker = col.replace("_weight", "")
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=ticker,
                stackgroup="weights",
                line=dict(width=0),
                fillcolor=palette[i % len(palette)],
            ), row=3, col=1)

        fig.update_layout(
            title=dict(
                text=f"Portfolio Backtest — {self.strategy.replace('_',' ').title()}",
                font=dict(size=16, color="#F8FAFC"),
            ),
            height=800,
            template="plotly_white",
            paper_bgcolor="#0A0E1A",
            plot_bgcolor="#0F1729",
            font=dict(color="#94A3B8"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#1E293B")
        fig.update_yaxes(showgrid=True, gridcolor="#1E293B")

        if save_path:
            ensure_dirs(project_path("results"))
            fig.write_html(str(save_path))
            logger.success(f"Portfolio chart saved → {save_path}")
        else:
            fig.show()

    def correlation_matrix(self) -> pd.DataFrame:
        """Return and print the return correlation matrix across tickers."""
        corr = self.returns_df.corr().round(3)
        print("\nReturn Correlation Matrix:")
        print(corr.to_string())
        return corr