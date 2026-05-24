"""
src/explainability.py
──────────────────────
SHAP-based model explainability for XGBoost predictions.

Answers the key interview question:
  "Why did your model predict UP/DOWN for this stock today?"

Features:
  - Global feature importance via SHAP summary plot
  - Local (per-prediction) waterfall explanation
  - SHAP dependence plots for top features
  - Force plots for individual predictions
  - Feature interaction detection

Usage:
    from src.explainability import SHAPExplainer

    ex = SHAPExplainer(model, X_train)
    ex.summary_plot(X_test, save_path='results/shap_summary.html')
    ex.explain_prediction(X_test.iloc[[-1]], ticker='RELIANCE.NS')
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import ensure_dirs


class SHAPExplainer:
    """
    SHAP explainability wrapper for tree-based models (XGBoost, RF).

    Parameters
    ----------
    model        : Fitted model object (XGBoostModel or pipeline)
    X_train      : Training features used to fit the model
    model_type   : 'tree' (XGBoost/RF) or 'linear'
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        model_type: str = "tree",
    ) -> None:
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError(
                "SHAP not installed. Run: pip install shap"
            )

        self.model_type   = model_type
        self.feature_names = list(X_train.columns)

        # Extract raw estimator from pipeline if needed
        if hasattr(model, "named_steps"):
            self.scaler    = model.named_steps.get("scaler")
            self.estimator = model.named_steps["model"]
        else:
            self.scaler    = None
            self.estimator = model

        # Scale training data
        X_scaled = self._scale(X_train)

        # Build SHAP explainer
        logger.info("Building SHAP TreeExplainer...")
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(self.estimator)
        else:
            self.explainer = shap.LinearExplainer(
                self.estimator,
                X_scaled,
                feature_names=self.feature_names,
            )

        # Compute SHAP values for training set (used as background)
        sample_size      = min(200, len(X_train))
        X_sample         = X_scaled.iloc[:sample_size] if hasattr(X_scaled, "iloc") else X_scaled[:sample_size]
        self.shap_values_train = self.explainer.shap_values(X_sample)
        self.X_train_scaled    = X_scaled
        logger.success("SHAP explainer ready")

    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaler if available, return DataFrame."""
        if self.scaler is not None:
            arr = self.scaler.transform(X)
            return pd.DataFrame(arr, index=X.index, columns=X.columns)
        return X

    # ══════════════════════════════════════════════════════════════════════════
    # Global explanations
    # ══════════════════════════════════════════════════════════════════════════

    def summary_plot(
        self,
        X_test: pd.DataFrame,
        max_display: int = 20,
        save_path: Optional[str | Path] = None,
    ) -> pd.Series:
        """
        SHAP summary plot — shows which features matter most globally
        and whether they push predictions up or down.

        Returns a Series of mean |SHAP| values (feature importances).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X_sc      = self._scale(X_test)
        shap_vals = self.explainer.shap_values(X_sc)

        # Handle multi-output (classification returns list)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 (UP)

        # Mean absolute SHAP = global importance
        importance = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=self.feature_names,
        ).sort_values(ascending=False)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0F1729")

        self.shap.summary_plot(
            shap_vals,
            X_sc,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Feature Importance", color="#F8FAFC", fontsize=14, pad=15)
        plt.tight_layout()

        if save_path:
            ensure_dirs(Path(save_path).parent)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                        facecolor="#0A0E1A")
            logger.success(f"SHAP summary plot saved → {save_path}")
        else:
            plt.show()
        plt.close()

        # Print top features
        print("\nTop 10 Features by Mean |SHAP|:")
        print(importance.head(10).to_string())
        return importance

    def bar_plot(
        self,
        X_test: pd.DataFrame,
        max_display: int = 15,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Simple bar chart of global SHAP importances."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X_sc      = self._scale(X_test)
        shap_vals = self.explainer.shap_values(X_sc)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        importance = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=self.feature_names,
        ).sort_values(ascending=False).head(max_display)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0F1729")

        bars = ax.barh(
            importance.index[::-1],
            importance.values[::-1],
            color="#3B82F6",
        )
        ax.set_xlabel("Mean |SHAP Value|", color="#94A3B8")
        ax.set_title("Global Feature Importance (SHAP)", color="#F8FAFC", fontsize=13)
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E293B")

        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                        facecolor="#0A0E1A")
            logger.success(f"SHAP bar plot saved → {save_path}")
        else:
            plt.show()
        plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Local explanations (per prediction)
    # ══════════════════════════════════════════════════════════════════════════

    def explain_prediction(
        self,
        X_row: pd.DataFrame,
        ticker: str = "Stock",
        save_path: Optional[str | Path] = None,
    ) -> dict:
        """
        Explain a SINGLE prediction with a waterfall chart.

        Shows exactly which features pushed the prediction UP or DOWN
        from the baseline (average prediction).

        Parameters
        ----------
        X_row    : Single-row DataFrame (the prediction to explain)
        ticker   : Display name for the chart title
        save_path: Save path for PNG (None = show inline)

        Returns
        -------
        dict with top positive and negative SHAP contributors
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X_sc      = self._scale(X_row)
        shap_vals = self.explainer.shap_values(X_sc)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_row = shap_vals[0]

        # Top contributors
        contribs = pd.Series(shap_row, index=self.feature_names)
        top_pos  = contribs.nlargest(5)
        top_neg  = contribs.nsmallest(5)

        # Waterfall chart
        top_n   = 12
        sorted_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
        features   = [self.feature_names[i] for i in sorted_idx]
        values     = shap_row[sorted_idx]
        colors     = ["#22C55E" if v > 0 else "#EF4444" for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0F1729")

        y_pos = range(len(features))
        ax.barh(y_pos, values, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, color="#94A3B8", fontsize=9)
        ax.set_xlabel("SHAP Value (contribution to prediction)", color="#94A3B8")
        ax.set_title(
            f"Prediction Explanation — {ticker}\n"
            f"(Green = pushes prediction UP, Red = pushes DOWN)",
            color="#F8FAFC", fontsize=12,
        )
        ax.axvline(0, color="#64748B", linewidth=0.8)
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E293B")

        plt.tight_layout()
        if save_path:
            ensure_dirs(Path(save_path).parent)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                        facecolor="#0A0E1A")
            logger.success(f"Waterfall plot saved → {save_path}")
        else:
            plt.show()
        plt.close()

        result = {
            "top_positive": top_pos.to_dict(),
            "top_negative": top_neg.to_dict(),
            "all_shap":     contribs.to_dict(),
        }

        print(f"\nPrediction explanation for {ticker}:")
        print("  Top features INCREASING prediction:")
        for f, v in top_pos.items():
            print(f"    {f:<35} +{v:.4f}")
        print("  Top features DECREASING prediction:")
        for f, v in top_neg.items():
            print(f"    {f:<35} {v:.4f}")

        return result

    def dependence_plot(
        self,
        X_test: pd.DataFrame,
        feature: str,
        interaction_feature: str = "auto",
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        SHAP dependence plot for a single feature.
        Shows how that feature's value affects predictions,
        coloured by a second feature to reveal interactions.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X_sc      = self._scale(X_test)
        shap_vals = self.explainer.shap_values(X_sc)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0A0E1A")

        self.shap.dependence_plot(
            feature,
            shap_vals,
            X_sc,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )
        ax.set_facecolor("#0F1729")
        ax.tick_params(colors="#94A3B8")
        ax.set_title(f"SHAP Dependence: {feature}", color="#F8FAFC")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E293B")

        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                        facecolor="#0A0E1A")
        else:
            plt.show()
        plt.close()