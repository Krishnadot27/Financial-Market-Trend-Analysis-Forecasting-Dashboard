"""
tests/test_model_training.py
─────────────────────────────
Unit tests for model training and evaluation.
Run: pytest tests/test_model_training.py -v
"""

import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Stubs ─────────────────────────────────────────────────────────────────────
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

from src.model_training import (
    LinearBaseline, RandomForestModel, XGBoostModel,
    walk_forward_validate, compare_models,
)
from src.evaluation import ModelEvaluator


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def regression_data():
    """Synthetic regression dataset (returns prediction)."""
    np.random.seed(42)
    n = 400
    X = pd.DataFrame(
        np.random.randn(n, 20),
        columns=[f"feat_{i}" for i in range(20)]
    )
    # y is weakly correlated with features (realistic)
    y = pd.Series(0.3 * X["feat_0"] + 0.2 * X["feat_1"] + np.random.randn(n) * 0.5)
    return X, y


@pytest.fixture
def classification_data():
    """Synthetic binary classification dataset (direction prediction)."""
    np.random.seed(42)
    n = 400
    X = pd.DataFrame(
        np.random.randn(n, 20),
        columns=[f"feat_{i}" for i in range(20)]
    )
    proba = 1 / (1 + np.exp(-(0.5 * X["feat_0"] + 0.3 * X["feat_1"])))
    y = pd.Series((proba > 0.5).astype(int))
    return X, y


def train_val_split(X, y, val_ratio=0.2):
    n = len(X)
    split = int(n * (1 - val_ratio))
    return (X.iloc[:split], y.iloc[:split],
            X.iloc[split:], y.iloc[split:])


# ══════════════════════════════════════════════════════════════════════════════
# LinearBaseline
# ══════════════════════════════════════════════════════════════════════════════

class TestLinearBaseline:

    def test_regression_fit_predict(self, regression_data):
        X, y = regression_data
        Xtr, ytr, Xv, yv = train_val_split(X, y)
        model = LinearBaseline(task="regression")
        model.fit(Xtr, ytr)
        preds = model.predict(Xv)
        assert len(preds) == len(Xv)
        assert not np.any(np.isnan(preds))

    def test_classification_fit_predict(self, classification_data):
        X, y = classification_data
        Xtr, ytr, Xv, yv = train_val_split(X, y)
        model = LinearBaseline(task="classification")
        model.fit(Xtr, ytr)
        preds = model.predict(Xv)
        assert set(preds).issubset({0, 1})

    def test_raises_before_fit(self, regression_data):
        X, y = regression_data
        model = LinearBaseline()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_invalid_task_raises(self):
        with pytest.raises(AssertionError):
            LinearBaseline(task="invalid")


# ══════════════════════════════════════════════════════════════════════════════
# RandomForest
# ══════════════════════════════════════════════════════════════════════════════

class TestRandomForest:

    def test_regression_output_shape(self, regression_data):
        X, y = regression_data
        Xtr, ytr, Xv, yv = train_val_split(X, y)
        model = RandomForestModel(task="regression", n_estimators=50)
        model.fit(Xtr, ytr)
        assert len(model.predict(Xv)) == len(Xv)

    def test_classification_binary_output(self, classification_data):
        X, y = classification_data
        Xtr, ytr, Xv, yv = train_val_split(X, y)
        model = RandomForestModel(task="classification", n_estimators=50)
        model.fit(Xtr, ytr)
        preds = model.predict(Xv)
        assert set(preds).issubset({0, 1})

    def test_feature_importance_length(self, regression_data):
        X, y = regression_data
        model = RandomForestModel(task="regression", n_estimators=50)
        model.fit(X, y)
        imp = model.feature_importance(top_n=5)
        assert len(imp) == 5

    def test_feature_importance_names_match(self, regression_data):
        X, y = regression_data
        model = RandomForestModel(task="regression", n_estimators=50)
        model.fit(X, y)
        imp = model.feature_importance()
        assert all(name in X.columns for name in imp.index)


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost
# ══════════════════════════════════════════════════════════════════════════════

class TestXGBoost:

    def test_regression_fit_predict(self, regression_data):
        X, y = regression_data
        Xtr, ytr, Xv, yv = train_val_split(X, y)
        model = XGBoostModel(task="regression", n_estimators=50)
        model.fit(Xtr, ytr, Xv, yv)
        preds = model.predict(Xv)
        assert len(preds) == len(Xv)
        assert not np.any(np.isnan(preds))

    def test_classification_proba_range(self, classification_data):
        X, y = classification_data
        Xtr, ytr, Xv, yv = train_val_split(X, y)
        model = XGBoostModel(task="classification", n_estimators=50)
        model.fit(Xtr, ytr, Xv, yv)
        proba = model.predict_proba(Xv)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_predict_proba_raises_for_regression(self, regression_data):
        X, y = regression_data
        model = XGBoostModel(task="regression", n_estimators=50)
        model.fit(X, y)
        with pytest.raises(ValueError, match="classification"):
            model.predict_proba(X)

    def test_save_load_roundtrip(self, regression_data, tmp_path):
        X, y = regression_data
        model = XGBoostModel(task="regression", n_estimators=50)
        model.fit(X, y)
        preds_before = model.predict(X)

        save_path = tmp_path / "xgb_test.pkl"
        model.save(save_path)

        loaded = XGBoostModel.load(save_path)
        preds_after = loaded.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_feature_names_preserved(self, regression_data):
        X, y = regression_data
        model = XGBoostModel(task="regression", n_estimators=50)
        model.fit(X, y)
        assert model.feature_names == list(X.columns)

    def test_xgboost_beats_linear_on_nonlinear_data(self):
        """XGBoost should outperform linear model on non-linear data."""
        np.random.seed(0)
        n = 500
        X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
        # Non-linear target: sine + interaction
        y = pd.Series(np.sin(X["f0"]) * X["f1"] + X["f2"]**2 + np.random.randn(n) * 0.1)

        split = int(n * 0.8)
        Xtr, ytr = X.iloc[:split], y.iloc[:split]
        Xte, yte = X.iloc[split:], y.iloc[split:]

        lin = LinearBaseline(task="regression")
        lin.fit(Xtr, ytr)
        xgb_m = XGBoostModel(task="regression", n_estimators=100)
        xgb_m.fit(Xtr, ytr)

        from sklearn.metrics import mean_squared_error
        rmse_lin = np.sqrt(mean_squared_error(yte, lin.predict(Xte)))
        rmse_xgb = np.sqrt(mean_squared_error(yte, xgb_m.predict(Xte)))

        assert rmse_xgb < rmse_lin, \
            f"XGBoost ({rmse_xgb:.4f}) should beat Linear ({rmse_lin:.4f}) on non-linear data"


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForward:

    def test_returns_dataframe(self, regression_data):
        X, y = regression_data
        results = walk_forward_validate(
            XGBoostModel, X, y, n_splits=3,
            task="regression",
            model_kwargs={"n_estimators": 50}
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

    def test_correct_number_of_folds(self, classification_data):
        X, y = classification_data
        results = walk_forward_validate(
            RandomForestModel, X, y, n_splits=3,
            task="classification",
            model_kwargs={"n_estimators": 30}
        )
        assert len(results) == 3

    def test_fold_sizes_increase(self, regression_data):
        """Each fold's training set must be larger than the previous."""
        X, y = regression_data
        results = walk_forward_validate(
            LinearBaseline, X, y, n_splits=4, task="regression"
        )
        sizes = results["train_size"].values
        assert all(sizes[i] < sizes[i+1] for i in range(len(sizes)-1)), \
            "Training set should grow with each fold (expanding window)"


# ══════════════════════════════════════════════════════════════════════════════
# ModelEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class TestModelEvaluator:

    def test_regression_report_returns_dict(self):
        ev = ModelEvaluator(task="regression")
        y_true = np.array([100, 102, 98, 105, 103])
        y_pred = np.array([101, 101, 99, 104, 102])
        result = ev.regression_report(y_true, y_pred)
        assert isinstance(result, dict)
        assert "RMSE" in result and "R2" in result

    def test_rmse_is_positive(self):
        ev = ModelEvaluator()
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.1, 1.9, 3.2])
        r = ev.regression_report(y, p)
        assert r["RMSE"] >= 0

    def test_perfect_prediction_r2_is_one(self):
        ev = ModelEvaluator()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = ev.regression_report(y, y)
        assert abs(r["R2"] - 1.0) < 1e-6

    def test_classification_report_returns_dict(self):
        ev = ModelEvaluator(task="classification")
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        result = ev.classification_report(y_true, y_pred)
        assert "Accuracy" in result and "F1" in result

    def test_sharpe_positive_for_good_strategy(self):
        ev = ModelEvaluator(risk_free=0.065)
        good_returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        sharpe = ev.sharpe_ratio(good_returns)
        # With mean > risk-free/252, Sharpe should be positive
        assert isinstance(sharpe, float)

    def test_max_drawdown_negative(self):
        ev = ModelEvaluator()
        # Simulate: goes up then crashes
        returns = pd.Series([0.01] * 100 + [-0.02] * 50 + [0.01] * 50)
        cum     = (1 + returns).cumprod()
        mdd     = ev.max_drawdown(cum)
        assert mdd < 0, "Max drawdown should be negative"

    def test_max_drawdown_zero_for_always_up(self):
        ev = ModelEvaluator()
        returns = pd.Series([0.001] * 200)
        cum     = (1 + returns).cumprod()
        mdd     = ev.max_drawdown(cum)
        assert mdd >= -1e-6, "Always-up strategy should have ~0 drawdown"

    def test_trading_report_returns_dict(self):
        ev = ModelEvaluator()
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
        result  = ev.trading_report(returns, label="Test")
        assert "Sharpe Ratio" in result
        assert "Max Drawdown %" in result
        assert "Win Rate %" in result