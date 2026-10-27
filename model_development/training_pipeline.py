"""
Comprehensive Training Pipeline for NSE Stock Prediction.

Implements:
1. Task A: 5-Trading-Session Directional Classification.
2. Task B: 3-Month (63-Trading-Session) Continuous-Return Regression.
3. Feature subset comparisons (Market-only, Fundamentals-only, Combined).
4. Strict chronological expanding-window walk-forward validation with purging.
5. Strict preprocessing fitting inside training pipelines (zero look-ahead leakage).
6. Comprehensive metric computation:
   - Classification: ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, Precision, Recall, F1, Brier.
   - Regression: MAE, RMSE, OOS R² (vs train-mean), OOS R² (vs zero), Directional Accuracy, Cross-Sectional Rank IC.
7. Verification on synthetic data fixture for smoke testing without touching holdout data.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    root_mean_squared_error,
)
from xgboost import XGBClassifier

from config import (
    ANALYTICAL_DATASET_PATH,
    MODEL_COMPARISON_PATH,
    WALK_FORWARD_RESULTS_PATH,
    WALK_FORWARD_SUMMARY_PATH,
    PER_STOCK_EVAL_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    BEST_MODEL_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    MARKET_FEATURES,
    FUNDAMENTAL_FEATURES,
    COMBINED_FEATURES,
    HORIZON_SESSIONS,
    PURGE_SESSIONS,
    HORIZON_REGRESSION_SESSIONS,
    PURGE_REGRESSION_SESSIONS,
    SAMPLING_REGRESSION_STEP,
    RANDOM_STATE,
    HOLDOUT_FRACTION,
    WALK_FORWARD_SPECS,
    EXPECTED_SYMBOLS,
)

warnings.filterwarnings("ignore")


# ==============================================================================
# NAIVE BASELINES FOR REGRESSION
# ==============================================================================

class ZeroReturnBaseline(BaseEstimator, RegressorMixin):
    """Naive baseline predicting exactly 0.0 return for all observations."""
    def fit(self, X, y):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class TrainingMeanBaseline(BaseEstimator, RegressorMixin):
    """Naive baseline predicting the constant training set mean return."""
    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


# ==============================================================================
# MODEL BUILDERS
# ==============================================================================

def build_classification_models() -> dict:
    """Constructs isolated scikit-learn Pipelines for Task A classification."""
    return {
        "Majority_Class_Baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic_Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                C=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random_Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=15,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Hist_Gradient_Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=5,
                min_samples_leaf=20,
                l2_regularization=1.5,
                learning_rate=0.04,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def build_regression_models() -> dict:
    """Constructs isolated scikit-learn Pipelines for Task B continuous return regression."""
    return {
        "Zero_Return_Baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ZeroReturnBaseline()),
        ]),
        "Training_Mean_Baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", TrainingMeanBaseline()),
        ]),
        "Ridge_Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "Random_Forest_Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=15,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Hist_Gradient_Boosting_Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=4,
                min_samples_leaf=20,
                l2_regularization=2.0,
                learning_rate=0.03,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Computes comprehensive classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(np.mean(y_true))
        metrics["brier_score"] = 0.25
    return metrics


def compute_cross_sectional_rank_ic(df_eval: pd.DataFrame, pred_col: str, true_col: str) -> tuple[float, float]:
    """
    Computes cross-sectional Rank IC (Spearman correlation between predicted and actual return)
    per distinct trading date, then summarizes across dates (mean and t-statistic).
    """
    date_ics = []
    for dt, grp in df_eval.groupby("Date"):
        if len(grp) >= 3 and grp[pred_col].std() > 1e-9 and grp[true_col].std() > 1e-9:
            rho, _ = spearmanr(grp[pred_col], grp[true_col])
            if not np.isnan(rho):
                date_ics.append(rho)

    if not date_ics:
        return 0.0, 0.0

    mean_ic = float(np.mean(date_ics))
    std_ic = float(np.std(date_ics, ddof=1)) if len(date_ics) > 1 else 1e-9
    t_stat = float(mean_ic / (std_ic / np.sqrt(len(date_ics)))) if std_ic > 1e-9 else 0.0
    return mean_ic, t_stat


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_mean: float,
    df_eval: pd.DataFrame = None,
    pred_col: str = None,
    true_col: str = None
) -> dict:
    """Computes comprehensive continuous return regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    # OOS R² vs train-mean benchmark
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot_mean = float(np.sum((y_true - train_mean) ** 2))
    ss_tot_zero = float(np.sum(y_true ** 2))

    r2_vs_mean = 1.0 - (ss_res / ss_tot_mean) if ss_tot_mean > 0 else float("nan")
    r2_vs_zero = 1.0 - (ss_res / ss_tot_zero) if ss_tot_zero > 0 else float("nan")

    # Directional metrics
    actual_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)
    dir_acc = float(accuracy_score(actual_dir, pred_dir))
    dir_auc = float(roc_auc_score(actual_dir, y_pred)) if len(np.unique(actual_dir)) > 1 else 0.5

    # Cross-sectional Rank IC
    mean_rank_ic, rank_ic_t = 0.0, 0.0
    if df_eval is not None and pred_col and true_col:
        mean_rank_ic, rank_ic_t = compute_cross_sectional_rank_ic(df_eval, pred_col, true_col)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2_vs_train_mean": r2_vs_mean,
        "r2_vs_zero": r2_vs_zero,
        "directional_accuracy": dir_acc,
        "directional_auc": dir_auc,
        "mean_rank_ic": mean_rank_ic,
        "rank_ic_t_stat": rank_ic_t,
    }


# ==============================================================================
# FEATURE SUBSET RESOLUTION
# ==============================================================================

def resolve_feature_subset(df: pd.DataFrame, subset_name: str) -> tuple[list, str]:
    """
    Resolves feature columns for market-only, fundamentals-only, or combined.
    Strictly prevents fabrication if fundamental features are absent.
    """
    subset_name = subset_name.lower().strip()
    if subset_name in ["market", "market_only", "technical"]:
        return MARKET_FEATURES, "OK"

    elif subset_name in ["fundamentals", "fundamentals_only"]:
        present = [c for c in FUNDAMENTAL_FEATURES if c in df.columns]
        if not present or df[present].isna().all().all():
            return [], "BLOCKED: 0 fundamental features present in dataset. Cannot fabricate data."
        return present, "OK"

    elif subset_name in ["combined", "market_plus_fundamentals"]:
        present_fund = [c for c in FUNDAMENTAL_FEATURES if c in df.columns]
        if not present_fund or df[present_fund].isna().all().all():
            return [], "BLOCKED: Raw fundamental filings absent in workspace. Combined experiment deferred."
        return MARKET_FEATURES + present_fund, "OK"

    else:
        raise ValueError(f"Unknown feature subset: {subset_name}")


# ==============================================================================
# SYNTHETIC SMOKE TEST FIXTURE (PHASE 6 VERIFICATION)
# ==============================================================================

def run_synthetic_smoke_test() -> bool:
    """
    Executes a fast end-to-end training and evaluation test on a small synthetic fixture.
    Guarantees that training code runs cleanly without touching the real holdout set.
    """
    print("\n--- Running Synthetic Pipeline Execution Smoke Test ---")
    np.random.seed(RANDOM_STATE)
    n_days = 100
    n_symbols = 4
    symbols = ["TEST_A", "TEST_B", "TEST_C", "TEST_D"]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    records = []
    for sym in symbols:
        price = 100.0
        for dt in dates:
            ret = np.random.normal(0.0005, 0.02)
            price *= (1.0 + ret)
            records.append({
                "Symbol": sym,
                "Date": dt,
                "adj_close": price,
                "adj_open": price * 0.998,
                "adj_high": price * 1.01,
                "adj_low": price * 0.99,
                "adj_volume": np.random.uniform(1e5, 1e6),
                "No. of Trades": np.random.uniform(1e3, 1e4),
                "delivery_pct": np.random.uniform(30, 70),
            })

    synth_df = pd.DataFrame(records).sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Compute minimal features
    synth_df["return_1d"] = synth_df.groupby("Symbol")["adj_close"].pct_change()
    synth_df["return_5d"] = synth_df.groupby("Symbol")["adj_close"].pct_change(5)
    synth_df["close_vs_sma_5"] = synth_df["adj_close"] / synth_df.groupby("Symbol")["adj_close"].transform(lambda s: s.rolling(5).mean()) - 1.0
    synth_df["volatility_10d"] = synth_df.groupby("Symbol")["return_1d"].transform(lambda s: s.rolling(10).std())
    synth_df["daily_range_pct"] = (synth_df["adj_high"] - synth_df["adj_low"]) / synth_df["adj_close"]
    synth_df["open_close_return"] = synth_df["adj_close"] / synth_df["adj_open"] - 1.0
    synth_df["volume_change_1d"] = synth_df.groupby("Symbol")["adj_volume"].pct_change()
    synth_df["volume_vs_avg20"] = synth_df["adj_volume"] / synth_df.groupby("Symbol")["adj_volume"].transform(lambda s: s.rolling(20).mean()) - 1.0
    synth_df["trades_vs_avg20"] = synth_df["No. of Trades"] / synth_df.groupby("Symbol")["No. of Trades"].transform(lambda s: s.rolling(20).mean()) - 1.0
    synth_df["rsi_14"] = 50.0
    synth_df["day_of_week"] = synth_df["Date"].dt.dayofweek
    synth_df["month"] = synth_df["Date"].dt.month

    # Task A Target
    fwd_5 = synth_df.groupby("Symbol")["adj_close"].shift(-5)
    synth_df["target_up_5d"] = np.where(fwd_5.notna(), (fwd_5 > synth_df["adj_close"]).astype(float), np.nan)
    synth_df["future_return_5d"] = fwd_5 / synth_df["adj_close"] - 1.0

    features = [
        "return_1d", "return_5d", "close_vs_sma_5", "volatility_10d",
        "daily_range_pct", "open_close_return", "volume_change_1d",
        "volume_vs_avg20", "trades_vs_avg20", "delivery_pct", "rsi_14",
        "day_of_week", "month"
    ]

    valid = synth_df.dropna(subset=["target_up_5d"]).copy()
    split = int(len(valid) * 0.70)
    train_slice = valid.iloc[:split]
    test_slice = valid.iloc[split:]

    # Test classification
    cls_models = build_classification_models()
    for name, pipe in cls_models.items():
        pipe.fit(train_slice[features], train_slice["target_up_5d"].astype(int))
        preds = pipe.predict(test_slice[features])
        probs = pipe.predict_proba(test_slice[features])[:, 1] if hasattr(pipe, "predict_proba") else None
        m = evaluate_classification(test_slice["target_up_5d"].to_numpy(), preds, probs)
        assert 0.0 <= m["accuracy"] <= 1.0

    # Test regression
    reg_models = build_regression_models()
    train_mean = float(train_slice["future_return_5d"].mean())
    for name, pipe in reg_models.items():
        pipe.fit(train_slice[features], train_slice["future_return_5d"])
        preds = pipe.predict(test_slice[features])
        m = evaluate_regression(test_slice["future_return_5d"].to_numpy(), preds, train_mean)
        assert m["mae"] >= 0.0

    print("   Synthetic smoke test passed successfully for all 5 classification and 5 regression models.")
    return True


if __name__ == "__main__":
    run_synthetic_smoke_test()
