"""
src/evaluation.py
──────────────────
Complete evaluation suite for stock prediction models.

Three evaluation layers:
  1. ML Metrics     — RMSE, MAE, R², Accuracy, F1, ROC-AUC
  2. Trading Metrics — Sharpe Ratio, Max Drawdown, Calmar, Win Rate
  3. Visualisations  — Predicted vs Actual, Confusion Matrix, ROC curve

Usage:
    from src.evaluation import ModelEvaluator
    ev = ModelEvaluator(task='regression')
    ev.regression_report(y_true, y_pred)
    ev.trading_report(strategy_returns)
    ev.plot_predictions(dates, y_true, y_pred, ticker='RELIANCE.NS')
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# ModelEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    Unified evaluation class for regression and classification tasks.

    Parameters
    ----------
    task        : 'regression' | 'classification'
    risk_free   : Annual risk-free rate (India G-Sec ~6.5%)
    trading_days: Trading days per year (252 for NSE)
    """

    def __init__(
        self,
        task:         str   = "regression",
        risk_free:    float = 0.065,
        trading_days: int   = 252,
    ) -> None:
        self.task         = task
        self.risk_free    = risk_free
        self.trading_days = trading_days

    # ══════════════════════════════════════════════════════════════════════════
    # ML Metrics
    # ══════════════════════════════════════════════════════════════════════════

    def regression_report(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        label:  str = "",
    ) -> dict:
        """
        Compute and print regression metrics.
        Returns dict for programmatic use.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(
            np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true)))
        ) * 100
        # Direction accuracy — did we predict the right sign?
        dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

        metrics = {
            "RMSE":      round(rmse, 4),
            "MAE":       round(mae, 4),
            "R2":        round(r2, 4),
            "MAPE_%":    round(mape, 2),
            "DirAcc_%":  round(dir_acc, 2),
        }

        title = f"Regression Metrics{' — ' + label if label else ''}"
        self._print_box(title, metrics)
        return metrics

    def classification_report(
        self,
        y_true:   np.ndarray | pd.Series,
        y_pred:   np.ndarray | pd.Series,
        y_proba:  Optional[np.ndarray | pd.Series] = None,
        label:    str = "",
    ) -> dict:
        """
        Compute and print classification metrics.
        y_proba: predicted probabilities for class 1 (UP) — needed for ROC-AUC.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, classification_report,
        )

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        metrics = {
            "Accuracy":  round(acc, 4),
            "Precision": round(prec, 4),
            "Recall":    round(rec, 4),
            "F1":        round(f1, 4),
        }

        if y_proba is not None:
            auc = roc_auc_score(y_true, np.array(y_proba))
            metrics["ROC_AUC"] = round(auc, 4)

        title = f"Classification Metrics{' — ' + label if label else ''}"
        self._print_box(title, metrics)

        # Detailed per-class report
        print("\nDetailed Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=["DOWN (0)", "UP (1)"],
            zero_division=0,
        ))
        return metrics

    # ══════════════════════════════════════════════════════════════════════════
    # Trading Metrics
    # ══════════════════════════════════════════════════════════════════════════

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Annualised Sharpe Ratio.
        India risk-free rate ~6.5% → daily = 6.5%/252
        Sharpe > 1.0 = good, > 2.0 = excellent
        """
        daily_rf   = self.risk_free / self.trading_days
        excess_ret = returns - daily_rf
        if excess_ret.std() < 1e-10:
            return 0.0
        return float((excess_ret.mean() / excess_ret.std()) * np.sqrt(self.trading_days))

    def max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Maximum peak-to-trough percentage loss.
        e.g. -0.15 means the strategy lost 15% from its peak at worst.
        """
        roll_max  = cumulative_returns.cummax()
        drawdown  = (cumulative_returns - roll_max) / roll_max.replace(0, np.nan)
        return float(drawdown.min())

    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calmar = Annualised Return / |Max Drawdown|
        Penalises strategies with large drawdowns.
        """
        annual_ret = (1 + returns.mean()) ** self.trading_days - 1
        cum        = (1 + returns).cumprod()
        mdd        = self.max_drawdown(cum)
        if abs(mdd) < 1e-10:
            return 0.0
        return float(annual_ret / abs(mdd))

    def trading_report(
        self,
        strategy_returns: pd.Series,
        label: str = "Strategy",
    ) -> dict:
        """
        Compute and print all trading performance metrics.

        Parameters
        ----------
        strategy_returns : daily return series (e.g. 0.012 = 1.2% day return)
        """
        cum         = (1 + strategy_returns).cumprod()
        total_ret   = float(cum.iloc[-1] - 1)
        annual_ret  = float((1 + strategy_returns.mean()) ** self.trading_days - 1)
        sharpe      = self.sharpe_ratio(strategy_returns)
        mdd         = self.max_drawdown(cum)
        calmar      = self.calmar_ratio(strategy_returns)
        win_rate    = float((strategy_returns > 0).mean())
        avg_win     = float(strategy_returns[strategy_returns > 0].mean()) if (strategy_returns > 0).any() else 0
        avg_loss    = float(strategy_returns[strategy_returns < 0].mean()) if (strategy_returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        metrics = {
            "Total Return %":   round(total_ret * 100, 2),
            "Annual Return %":  round(annual_ret * 100, 2),
            "Sharpe Ratio":     round(sharpe, 3),
            "Max Drawdown %":   round(mdd * 100, 2),
            "Calmar Ratio":     round(calmar, 3),
            "Win Rate %":       round(win_rate * 100, 1),
            "Profit Factor":    round(profit_factor, 2),
            "Avg Win %":        round(avg_win * 100, 3),
            "Avg Loss %":       round(avg_loss * 100, 3),
        }

        self._print_box(f"Trading Performance — {label}", metrics)
        return metrics

    def compare_strategies(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_label:   str = "ML Strategy",
        benchmark_label:  str = "Buy & Hold",
    ) -> pd.DataFrame:
        """
        Side-by-side comparison of strategy vs benchmark.
        Returns a DataFrame you can add to your report/README.
        """
        strat_metrics = self.trading_report(strategy_returns, strategy_label)
        bench_metrics = self.trading_report(benchmark_returns, benchmark_label)

        df = pd.DataFrame({
            strategy_label: strat_metrics,
            benchmark_label: bench_metrics,
        })

        # Alpha
        alpha = strat_metrics["Annual Return %"] - bench_metrics["Annual Return %"]
        print(f"\n  Alpha (vs {benchmark_label}): {alpha:+.2f}%")
        if strat_metrics["Sharpe Ratio"] > bench_metrics["Sharpe Ratio"]:
            print("  ✔ Strategy has BETTER risk-adjusted returns")
        else:
            print("  ✘ Strategy has WORSE risk-adjusted returns than benchmark")

        return df

    # ══════════════════════════════════════════════════════════════════════════
    # Plotting (requires matplotlib/plotly)
    # ══════════════════════════════════════════════════════════════════════════

    def plot_predictions(
        self,
        dates:   pd.DatetimeIndex,
        y_true:  np.ndarray | pd.Series,
        y_pred:  np.ndarray | pd.Series,
        ticker:  str = "Stock",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot predicted vs actual values over time."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed — skipping plot")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=y_true,
            name="Actual", line=dict(color="#1E293B", width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=y_pred,
            name="Predicted", line=dict(color="#2563EB", width=1.5, dash="dot")
        ))
        fig.update_layout(
            title=f"{ticker} — Predicted vs Actual",
            xaxis_title="Date",
            yaxis_title="Price / Return",
            template="plotly_white",
            height=450,
        )
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved → {save_path}")
        else:
            fig.show()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        ticker: str = "Stock",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix for direction prediction."""
        try:
            import plotly.figure_factory as ff
            from sklearn.metrics import confusion_matrix
        except ImportError:
            logger.warning("plotly or sklearn not installed — skipping plot")
            return

        cm = confusion_matrix(y_true, y_pred)
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=["Pred DOWN", "Pred UP"],
            y=["True DOWN", "True UP"],
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(
            title=f"{ticker} — Direction Prediction Confusion Matrix",
            template="plotly_white",
        )
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    # ══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _print_box(title: str, metrics: dict) -> None:
        width = 50
        print(f"\n{'═' * width}")
        print(f"  {title}")
        print(f"{'─' * width}")
        for k, v in metrics.items():
            print(f"  {k:<22} {v}")
        print(f"{'═' * width}")