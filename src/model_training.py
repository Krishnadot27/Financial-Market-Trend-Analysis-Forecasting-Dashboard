"""
src/model_training.py
──────────────────────
Production-grade model training for NSE stock prediction.

Models implemented:
  1. LinearRegression  — baseline (regression)
  2. RandomForest      — regression + classification
  3. XGBoost           — primary model (regression + classification)

Each model class follows the same interface:
  .fit(X_train, y_train, X_val, y_val)
  .predict(X)
  .predict_proba(X)   ← classification only
  .save(path)
  .load(path)

Pipeline features:
  - StandardScaler inside Pipeline (prevents leakage)
  - TimeSeriesSplit cross-validation
  - GridSearchCV hyperparameter tuning
  - Feature importance extraction
  - Model serialisation with joblib
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.utils import ensure_dirs, project_path, set_seed, timer


# ══════════════════════════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════════════════════════

class BaseStockModel:
    """
    Shared interface for all stock prediction models.
    Subclasses implement _build_pipeline() and optionally tune().
    """

    def __init__(self, task: str = "regression", seed: int = 42) -> None:
        assert task in ("regression", "classification"), \
            "task must be 'regression' or 'classification'"
        self.task     = task
        self.seed     = seed
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        self.feature_names: list[str] = []
        self.train_time: float = 0.0

    # ── Abstract ──────────────────────────────────────────────────────────────
    def _build_pipeline(self) -> Pipeline:
        raise NotImplementedError

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
    ) -> "BaseStockModel":
        self.feature_names = list(X_train.columns)
        if self.pipeline is None:
            self.pipeline = self._build_pipeline()

        logger.info(f"Training {self.__class__.__name__}  "
                    f"[task={self.task}, features={X_train.shape[1]}, "
                    f"train_rows={len(X_train)}]")
        t0 = time.perf_counter()
        self.pipeline.fit(X_train, y_train)
        self.train_time = time.perf_counter() - t0
        self.is_fitted  = True

        # Quick validation score
        if X_val is not None and y_val is not None:
            self._log_val_score(X_val, y_val)

        logger.success(f"{self.__class__.__name__} trained in {self.train_time:.1f}s")
        return self

    def _log_val_score(self, X_val, y_val):
        y_pred = self.predict(X_val)
        if self.task == "regression":
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2   = r2_score(y_val, y_pred)
            logger.info(f"  Val RMSE={rmse:.4f}  R²={r2:.4f}")
        else:
            acc = accuracy_score(y_val, y_pred)
            f1  = f1_score(y_val, y_pred, zero_division=0)
            logger.info(f"  Val Acc={acc:.4f}  F1={f1:.4f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError(f"{self.__class__.__name__} has no predict_proba")
        return self.pipeline.predict_proba(X)[:, 1]

    # ── Feature importance ────────────────────────────────────────────────────
    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """Return top_n features by importance (model-specific)."""
        self._check_fitted()
        estimator = self.pipeline.named_steps["model"]

        if hasattr(estimator, "feature_importances_"):
            imp = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            imp = np.abs(estimator.coef_).flatten()
        else:
            raise AttributeError("Model has no feature_importances_ or coef_")

        series = pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
        return series.head(top_n)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: Optional[str | Path] = None) -> Path:
        self._check_fitted()
        if path is None:
            name = f"{self.__class__.__name__.lower()}_{self.task}.pkl"
            path = project_path("models") / name
        path = Path(path)
        ensure_dirs(path.parent)
        joblib.dump({
            "pipeline":      self.pipeline,
            "task":          self.task,
            "feature_names": self.feature_names,
            "train_time":    self.train_time,
        }, path)
        logger.success(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseStockModel":
        data = joblib.load(path)
        obj  = cls.__new__(cls)
        obj.pipeline      = data["pipeline"]
        obj.task          = data["task"]
        obj.feature_names = data["feature_names"]
        obj.train_time    = data["train_time"]
        obj.is_fitted     = True
        logger.info(f"Model loaded from {path}")
        return obj

    def _check_fitted(self):
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Linear Baseline
# ══════════════════════════════════════════════════════════════════════════════

class LinearBaseline(BaseStockModel):
    """
    Linear Regression / Logistic Regression baseline.

    Purpose: Establish a minimum performance floor.
    If your fancy model can't beat this, your features are wrong.
    """

    def _build_pipeline(self) -> Pipeline:
        if self.task == "regression":
            estimator = LinearRegression()
        else:
            estimator = LogisticRegression(
                max_iter=1000, C=1.0,
                random_state=self.seed, n_jobs=-1,
            )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  estimator),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# 2. Random Forest
# ══════════════════════════════════════════════════════════════════════════════

class RandomForestModel(BaseStockModel):
    """
    Random Forest — strong non-linear baseline.

    Advantages over linear:
      - Handles feature interactions automatically
      - Built-in feature importance
      - Robust to outliers and scale
    """

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 200,
        max_depth: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__(task, seed)
        self.n_estimators = n_estimators
        self.max_depth    = max_depth

    def _build_pipeline(self) -> Pipeline:
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=5,      # Prevents overfitting on financial data
            max_features="sqrt",
            random_state=self.seed,
            n_jobs=-1,
        )
        if self.task == "regression":
            estimator = RandomForestRegressor(**params)
        else:
            estimator = RandomForestClassifier(**params)

        return Pipeline([
            ("scaler", StandardScaler()),   # RF doesn't need scaling but
            ("model",  estimator),          # keeps interface consistent
        ])

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 3,
    ) -> "RandomForestModel":
        """Grid search on a small param grid — fast enough for RF."""
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [6, 8, 10],
            "model__min_samples_leaf": [3, 5],
        }
        scoring = "neg_root_mean_squared_error" if self.task == "regression" else "f1"
        tscv    = TimeSeriesSplit(n_splits=n_splits, gap=1)
        gs      = GridSearchCV(
            self._build_pipeline(), param_grid,
            cv=tscv, scoring=scoring, n_jobs=-1, verbose=0,
        )
        gs.fit(X_train, y_train)
        logger.info(f"RF best params: {gs.best_params_}  score={gs.best_score_:.4f}")
        self.pipeline  = gs.best_estimator_
        self.is_fitted = True
        self.feature_names = list(X_train.columns)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 3. XGBoost  (Primary Model)
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostModel(BaseStockModel):
    """
    XGBoost — primary production model.

    Why XGBoost for NSE daily data:
      - Best-in-class on tabular/feature-engineered data
      - Handles missing values natively
      - Built-in L1/L2 regularisation prevents overfitting
      - Feature importance + SHAP compatible
      - Much faster to train than LSTM on daily frequency

    Two modes:
      task='regression'     → XGBRegressor  → predicts next-day return/price
      task='classification' → XGBClassifier → predicts up/down direction
    """

    def __init__(
        self,
        task:              str   = "regression",
        n_estimators:      int   = 500,
        learning_rate:     float = 0.05,
        max_depth:         int   = 6,
        subsample:         float = 0.8,
        colsample_bytree:  float = 0.8,
        reg_alpha:         float = 0.1,
        reg_lambda:        float = 1.0,
        early_stopping:    int   = 50,
        seed:              int   = 42,
    ) -> None:
        super().__init__(task, seed)
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha        = reg_alpha
        self.reg_lambda       = reg_lambda
        self.early_stopping   = early_stopping

    def _get_xgb_estimator(self):
        common = dict(
            n_estimators      = self.n_estimators,
            learning_rate     = self.learning_rate,
            max_depth         = self.max_depth,
            subsample         = self.subsample,
            colsample_bytree  = self.colsample_bytree,
            reg_alpha         = self.reg_alpha,
            reg_lambda        = self.reg_lambda,
            random_state      = self.seed,
            n_jobs            = -1,
            verbosity         = 0,
        )
        if self.task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", **common)
        else:
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                **common,
            )

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  self._get_xgb_estimator()),
        ])

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
    ) -> "XGBoostModel":
        """
        Fit with optional early stopping on validation set.
        Early stopping prevents overfitting by halting when val score
        stops improving for `early_stopping` consecutive rounds.
        """
        self.feature_names = list(X_train.columns)
        self.pipeline      = self._build_pipeline()

        # Fit scaler on train only
        scaler  = self.pipeline.named_steps["scaler"]
        X_tr_sc = scaler.fit_transform(X_train)

        estimator = self.pipeline.named_steps["model"]

        logger.info(
            f"Training XGBoost [{self.task}]  "
            f"features={X_train.shape[1]}  rows={len(X_train)}"
        )
        t0 = time.perf_counter()

        if X_val is not None and y_val is not None:
            X_val_sc = scaler.transform(X_val)
            estimator.fit(
                X_tr_sc, y_train,
                eval_set=[(X_tr_sc, y_train), (X_val_sc, y_val)],
                verbose=False,
            )
        else:
            estimator.fit(X_tr_sc, y_train)

        self.train_time = time.perf_counter() - t0
        self.is_fitted  = True

        if X_val is not None and y_val is not None:
            self._log_val_score(X_val, y_val)

        logger.success(
            f"XGBoost trained in {self.train_time:.1f}s  "
            f"best_iteration={getattr(estimator, 'best_iteration', 'N/A')}"
        )
        return self

    @timer
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 5,
        quick: bool = True,
    ) -> "XGBoostModel":
        """
        Walk-forward hyperparameter tuning with TimeSeriesSplit.

        quick=True  → small grid, ~5 min
        quick=False → full grid, ~30 min
        """
        if quick:
            param_grid = {
                "model__max_depth":       [4, 6],
                "model__learning_rate":   [0.05, 0.1],
                "model__n_estimators":    [200, 400],
                "model__subsample":       [0.8],
                "model__colsample_bytree":[0.8],
            }
        else:
            param_grid = {
                "model__max_depth":        [4, 6, 8],
                "model__learning_rate":    [0.01, 0.05, 0.1],
                "model__n_estimators":     [200, 400, 600],
                "model__subsample":        [0.7, 0.8, 0.9],
                "model__colsample_bytree": [0.7, 0.8, 0.9],
                "model__reg_alpha":        [0.0, 0.1, 0.5],
            }

        scoring = "neg_root_mean_squared_error" if self.task == "regression" else "f1"
        tscv    = TimeSeriesSplit(n_splits=n_splits, gap=1)

        logger.info(
            f"Tuning XGBoost [{self.task}]  "
            f"grid_size={np.prod([len(v) for v in param_grid.values()])}  "
            f"cv_splits={n_splits}"
        )

        gs = GridSearchCV(
            self._build_pipeline(),
            param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        gs.fit(X_train, y_train)

        logger.success(
            f"Best params : {gs.best_params_}\n"
            f"Best CV score: {gs.best_score_:.4f}"
        )
        self.pipeline      = gs.best_estimator_
        self.is_fitted     = True
        self.feature_names = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the scaler fitted on training data."""
        self._check_fitted()
        scaler    = self.pipeline.named_steps["scaler"]
        estimator = self.pipeline.named_steps["model"]
        X_sc      = scaler.transform(X)
        return estimator.predict(X_sc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of class 1 (UP) — classification only."""
        self._check_fitted()
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification task")
        scaler    = self.pipeline.named_steps["scaler"]
        estimator = self.pipeline.named_steps["model"]
        X_sc      = scaler.transform(X)
        return estimator.predict_proba(X_sc)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(
    model_cls,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    task: str = "regression",
    model_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation.

    Unlike k-fold, each fold:
      - trains on ALL data up to fold boundary
      - validates on the NEXT chronological chunk
      - never shuffles

    Returns a DataFrame with per-fold metrics.

    Example
    -------
    results = walk_forward_validate(XGBoostModel, X, y, task='regression')
    print(results)
    """
    model_kwargs = model_kwargs or {}
    tscv         = TimeSeriesSplit(n_splits=n_splits, gap=1)
    fold_results = []

    logger.info(
        f"Walk-forward CV: {n_splits} folds  "
        f"model={model_cls.__name__}  task={task}"
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_cls(task=task, **model_kwargs)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        if task == "regression":
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2   = r2_score(y_val, y_pred)
            fold_results.append({
                "fold": fold,
                "train_size": len(X_tr),
                "val_size":   len(X_val),
                "RMSE": round(rmse, 4),
                "R2":   round(r2, 4),
            })
            logger.info(f"  Fold {fold}: RMSE={rmse:.4f}  R²={r2:.4f}")
        else:
            acc = accuracy_score(y_val, y_pred)
            f1  = f1_score(y_val, y_pred, zero_division=0)
            fold_results.append({
                "fold": fold,
                "train_size": len(X_tr),
                "val_size":   len(X_val),
                "Accuracy": round(acc, 4),
                "F1":       round(f1, 4),
            })
            logger.info(f"  Fold {fold}: Acc={acc:.4f}  F1={f1:.4f}")

    results_df = pd.DataFrame(fold_results)

    # Summary row
    numeric_cols = results_df.select_dtypes(include=np.number).columns
    numeric_cols = [c for c in numeric_cols if c not in ["fold", "train_size", "val_size"]]
    logger.info(f"\nCV Summary:\n{results_df[['fold'] + numeric_cols].to_string(index=False)}")
    logger.info(f"Mean: { {c: round(results_df[c].mean(),4) for c in numeric_cols} }")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# Model comparison helper
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    task:    str = "regression",
) -> pd.DataFrame:
    """
    Train all three models and compare test-set performance side by side.

    Returns a DataFrame with one row per model and metric columns.
    """
    models = {
        "LinearBaseline": LinearBaseline(task=task),
        "RandomForest":   RandomForestModel(task=task),
        "XGBoost":        XGBoostModel(task=task),
    }

    rows = []
    for name, model in models.items():
        logger.info(f"\n── {name} ──")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "regression":
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = np.mean(np.abs(y_test - y_pred))
            r2   = r2_score(y_test, y_pred)
            rows.append({"Model": name, "RMSE": round(rmse, 4),
                         "MAE": round(mae, 4), "R2": round(r2, 4)})
        else:
            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            rows.append({"Model": name, "Accuracy": round(acc, 4),
                         "F1": round(f1, 4)})

    df = pd.DataFrame(rows).set_index("Model")
    print(f"\n{'═'*50}")
    print("  Model Comparison (Test Set)")
    print(f"{'═'*50}")
    print(df.to_string())
    print(f"{'═'*50}\n")
    return df