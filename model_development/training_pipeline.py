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
import argparse
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
import xgboost

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
    confusion_matrix,
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
    PREDICTIONS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
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

# ==============================================================================
# FORMAL EXPERIMENT EXECUTION ENGINE
# ==============================================================================

def run_training_experiment(
    task: str = "classification",
    feature_subset: str = "market",
    evaluate_holdout: bool = False,
    data_path: Path = ANALYTICAL_DATASET_PATH,
) -> dict:
    """
    Executes a formal, leakage-free training experiment.

    Parameters:
    - task: 'classification' (Task A, 5d direction) or 'regression' (Task B, 63d continuous return)
    - feature_subset: 'market', 'fundamentals', or 'combined'
    - evaluate_holdout: If False, keeps the final 15% holdout untouched.
    - data_path: Path to the prepared analytical dataset.
    """
    print("=" * 76)
    print(f"EXECUTING TRAINING EXPERIMENT: Task={task.upper()}, Subset={feature_subset.upper()}")
    print("=" * 76)

    if not data_path.exists():
        raise FileNotFoundError(f"Analytical dataset not found at: {data_path}. Run data_pipeline.py first.")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # Resolve features
    features, subset_status = resolve_feature_subset(df, feature_subset)
    if subset_status != "OK":
        print(f"\n[BLOCKED] {subset_status}")
        return {"status": "BLOCKED", "reason": subset_status}

    # Task configuration
    if task == "classification":
        target_col = "target_up_5d"
        purge_sessions = PURGE_SESSIONS
        models = build_classification_models()
        is_classification = True
    elif task == "regression":
        target_col = "future_return_63d" if "future_return_63d" in df.columns else "future_return_5d"
        purge_sessions = PURGE_REGRESSION_SESSIONS if target_col == "future_return_63d" else PURGE_SESSIONS
        models = build_regression_models()
        is_classification = False
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'classification' or 'regression'.")

    # Filter labeled data
    labeled_df = df.dropna(subset=[target_col]).copy()
    if is_classification:
        labeled_df[target_col] = labeled_df[target_col].astype(int)

    dates = np.array(sorted(labeled_df["Date"].unique()))
    n_dates = len(dates)

    # Chronological partition: 85% development, 15% holdout
    holdout_split_idx = int(n_dates * (1.0 - HOLDOUT_FRACTION))
    holdout_start_idx = holdout_split_idx + purge_sessions

    dev_dates = dates[:holdout_split_idx]
    holdout_dates = dates[holdout_start_idx:]

    dev_data = labeled_df[labeled_df["Date"].isin(dev_dates)].copy()
    holdout_data = labeled_df[labeled_df["Date"].isin(holdout_dates)].copy()

    print(f"Total labeled rows: {len(labeled_df):,} across {n_dates} dates.")
    print(f"Development Set: {len(dev_data):,} rows ({dev_dates[0].strftime('%Y-%m-%d')} to {dev_dates[-1].strftime('%Y-%m-%d')})")
    print(f"Purge Window: {purge_sessions} trading sessions")
    print(f"Holdout Set: {len(holdout_data):,} rows ({holdout_dates[0].strftime('%Y-%m-%d')} to {holdout_dates[-1].strftime('%Y-%m-%d')})")

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Walk-Forward Cross-Validation on Development Set
    # --------------------------------------------------------------------------
    print("\n--- Running Chronological Walk-Forward Cross-Validation ---")
    n_dev_dates = len(dev_dates)
    wf_records = []
    dev_fold_preds = []

    for fold_idx, (train_pct, val_pct) in enumerate(WALK_FORWARD_SPECS, 1):
        tr_end = int(n_dev_dates * train_pct)
        vl_start = tr_end + purge_sessions
        vl_end = int(n_dev_dates * val_pct)

        tr_dates = dev_dates[:tr_end]
        vl_dates = dev_dates[vl_start:vl_end]

        tr_fold = dev_data[dev_data["Date"].isin(tr_dates)]
        vl_fold = dev_data[dev_data["Date"].isin(vl_dates)]

        X_tr = tr_fold[features]
        y_tr = tr_fold[target_col]
        X_vl = vl_fold[features]
        y_vl = vl_fold[target_col]

        fold_models = build_classification_models() if is_classification else build_regression_models()
        tr_mean_val = float(y_tr.mean())

        fold_preds = pd.DataFrame({
            "fold": fold_idx,
            "Date": vl_fold["Date"].dt.strftime("%Y-%m-%d").values,
            "Symbol": vl_fold["Symbol"].values,
            "actual": y_vl.values,
        })

        for m_name, pipe in fold_models.items():
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_vl)
            fold_preds[f"{m_name}_pred"] = preds

            if is_classification:
                probs = pipe.predict_proba(X_vl)[:, 1] if hasattr(pipe, "predict_proba") else None
                if probs is not None:
                    fold_preds[f"{m_name}_prob"] = probs
                m = evaluate_classification(y_vl.to_numpy(), preds, probs)
            else:
                vl_fold_copy = vl_fold.copy()
                vl_fold_copy["_pred"] = preds
                m = evaluate_regression(y_vl.to_numpy(), preds, tr_mean_val, vl_fold_copy, "_pred", target_col)

            m["fold"] = fold_idx
            m["model"] = m_name
            m["train_rows"] = len(tr_fold)
            m["val_rows"] = len(vl_fold)
            wf_records.append(m)

        dev_fold_preds.append(fold_preds)

    wf_df = pd.DataFrame(wf_records)
    dev_preds_df = pd.concat(dev_fold_preds, ignore_index=True)

    # --------------------------------------------------------------------------
    # Summarize and Select Best Architecture
    # --------------------------------------------------------------------------
    if is_classification:
        summary_rows = []
        for m_name, grp in wf_df.groupby("model", sort=False):
            summary_rows.append({
                "model": m_name,
                "roc_auc_mean": grp["roc_auc"].mean(),
                "roc_auc_std": grp["roc_auc"].std(),
                "balanced_accuracy_mean": grp["balanced_accuracy"].mean(),
                "balanced_accuracy_std": grp["balanced_accuracy"].std(),
                "accuracy_mean": grp["accuracy"].mean(),
                "precision_mean": grp["precision"].mean(),
                "recall_mean": grp["recall"].mean(),
                "f1_mean": grp["f1"].mean(),
                "brier_score_mean": grp["brier_score"].mean(),
                "total_tn": int(grp["tn"].sum()),
                "total_fp": int(grp["fp"].sum()),
                "total_fn": int(grp["fn"].sum()),
                "total_tp": int(grp["tp"].sum()),
            })
        summary_df = pd.DataFrame(summary_rows)

        mean_scores = wf_df[wf_df["model"] != "Majority_Class_Baseline"].groupby("model")["roc_auc"].mean()
        best_model_name = mean_scores.idxmax()

        print("\n" + "=" * 95)
        print("TASK A: WALK-FORWARD CROSS-VALIDATION SUMMARY (CLASSIFICATION)")
        print("=" * 95)
        header = f"{'Model':30s} | {'ROC-AUC':17s} | {'Bal Acc':17s} | {'Precision':9s} | {'Recall':9s} | {'F1':9s}"
        print(header)
        print("-" * len(header))
        for _, r in summary_df.iterrows():
            auc_str = f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}"
            bacc_str = f"{r['balanced_accuracy_mean']:.4f} +/- {r['balanced_accuracy_std']:.4f}"
            print(f"{r['model']:30s} | {auc_str:17s} | {bacc_str:17s} | {r['precision_mean']:.4f}    | {r['recall_mean']:.4f}    | {r['f1_mean']:.4f}")

        print("\nFold-by-Fold Breakdown (ROC-AUC / Balanced Acc):")
        for f_idx in sorted(wf_df["fold"].unique()):
            print(f"  Fold {f_idx}:")
            for _, r in wf_df[wf_df["fold"] == f_idx].iterrows():
                print(f"    {r['model']:30s} | ROC-AUC: {r['roc_auc']:.4f} | Bal Acc: {r['balanced_accuracy']:.4f} | F1: {r['f1']:.4f} | [TN:{r['tn']} FP:{r['fp']} FN:{r['fn']} TP:{r['tp']}]")

        print(f"\nSelected Best Architecture: {best_model_name} (Mean WF ROC-AUC: {mean_scores[best_model_name]:.4f})")

    else:
        summary_rows = []
        for m_name, grp in wf_df.groupby("model", sort=False):
            summary_rows.append({
                "model": m_name,
                "mae_mean": grp["mae"].mean(),
                "mae_std": grp["mae"].std(),
                "rmse_mean": grp["rmse"].mean(),
                "rmse_std": grp["rmse"].std(),
                "r2_vs_train_mean": grp["r2_vs_train_mean"].mean(),
                "r2_vs_zero": grp["r2_vs_zero"].mean(),
                "directional_accuracy_mean": grp["directional_accuracy"].mean(),
                "directional_auc_mean": grp["directional_auc"].mean(),
                "mean_rank_ic": grp["mean_rank_ic"].mean(),
                "rank_ic_t_stat": grp["rank_ic_t_stat"].mean(),
            })
        summary_df = pd.DataFrame(summary_rows)

        mean_scores = wf_df[~wf_df["model"].str.contains("Baseline")].groupby("model")["mae"].mean()
        best_model_name = mean_scores.idxmin()

        print("\n" + "=" * 115)
        print("TASK B: WALK-FORWARD CROSS-VALIDATION SUMMARY (REGRESSION)")
        print("=" * 115)
        header = f"{'Model':32s} | {'MAE':17s} | {'RMSE':17s} | {'OOS R²(Mean)':12s} | {'Dir Acc':9s} | {'Rank IC':9s}"
        print(header)
        print("-" * len(header))
        for _, r in summary_df.iterrows():
            mae_str = f"{r['mae_mean']:.4f} +/- {r['mae_std']:.4f}"
            rmse_str = f"{r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f}"
            print(f"{r['model']:32s} | {mae_str:17s} | {rmse_str:17s} | {r['r2_vs_train_mean']:12.4f} | {r['directional_accuracy_mean']:.4f}    | {r['mean_rank_ic']:.4f}")

        print("\nFold-by-Fold Breakdown (MAE / OOS R² vs Mean / Rank IC):")
        for f_idx in sorted(wf_df["fold"].unique()):
            print(f"  Fold {f_idx}:")
            for _, r in wf_df[wf_df["fold"] == f_idx].iterrows():
                print(f"    {r['model']:32s} | MAE: {r['mae']:.4f} | RMSE: {r['rmse']:.4f} | OOS R²: {r['r2_vs_train_mean']:.4f} | Rank IC: {r['mean_rank_ic']:.4f} (t={r['rank_ic_t_stat']:.2f})")

        print(f"\nSelected Best Architecture: {best_model_name} (Mean WF MAE: {mean_scores[best_model_name]:.4f})")

    # --------------------------------------------------------------------------
    # Save Real Results to Established Directories
    # --------------------------------------------------------------------------
    for d in [REPORTS_DIR, PREDICTIONS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Per-observation development predictions
    dev_preds_path = PREDICTIONS_DIR / f"development_{task}_predictions.csv"
    dev_preds_df.to_csv(dev_preds_path, index=False)
    print(f"\n[ARTIFACT] Saved per-observation development predictions ({len(dev_preds_df):,} rows) to: {dev_preds_path}")

    # 2. Fold-wise results
    wf_results_path = REPORTS_DIR / f"walk_forward_{task}_results.csv"
    wf_df_to_save = wf_df.copy()
    if "confusion_matrix" in wf_df_to_save.columns:
        wf_df_to_save = wf_df_to_save.drop(columns=["confusion_matrix"])
    wf_df_to_save.to_csv(wf_results_path, index=False)
    print(f"[ARTIFACT] Saved fold-wise results to: {wf_results_path}")

    # 3. Model summary metrics
    summary_path = REPORTS_DIR / f"walk_forward_{task}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[ARTIFACT] Saved model summary metrics to: {summary_path}")

    # 4. Experiment configuration and provenance metadata JSON
    exp_config = {
        "task": task,
        "feature_subset": feature_subset,
        "features": features,
        "n_features": len(features),
        "dataset_path": str(data_path),
        "total_dataset_rows": len(df),
        "labeled_rows": len(labeled_df),
        "symbols": EXPECTED_SYMBOLS,
        "n_symbols": len(EXPECTED_SYMBOLS),
        "development_period": {
            "start_date": dev_dates[0].strftime("%Y-%m-%d"),
            "end_date": dev_dates[-1].strftime("%Y-%m-%d"),
            "n_dates": len(dev_dates),
            "n_rows": len(dev_data),
        },
        "purge_sessions": purge_sessions,
        "holdout_period": {
            "start_date": holdout_dates[0].strftime("%Y-%m-%d"),
            "end_date": holdout_dates[-1].strftime("%Y-%m-%d"),
            "n_dates": len(holdout_dates),
            "n_rows": len(holdout_data),
            "evaluated": evaluate_holdout,
        },
        "walk_forward_specs": WALK_FORWARD_SPECS,
        "random_state": RANDOM_STATE,
        "selected_best_model": best_model_name,
        "selection_metric": "roc_auc" if is_classification else "mae",
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": getattr(xgboost, "__version__", "unknown"),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "summary_metrics": summary_df.to_dict(orient="records"),
    }
    exp_config_path = REPORTS_DIR / f"experiment_config_{task}.json"
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"[ARTIFACT] Saved experiment configuration JSON to: {exp_config_path}")

    # 5. Fit best model on the full development set and serialize
    best_pipe = models[best_model_name]
    best_pipe.fit(dev_data[features], dev_data[target_col])
    best_dev_model_path = MODELS_DIR / f"{task}_best_dev_model.joblib"
    joblib.dump(best_pipe, best_dev_model_path)
    print(f"[ARTIFACT] Fitted best model ({best_model_name}) on full Development Set ({len(dev_data):,} rows) and saved to: {best_dev_model_path}")

    holdout_results = None
    if evaluate_holdout:
        print("\n--- Evaluating on Sequestered Final Holdout Window ---")
        X_holdout = holdout_data[features]
        y_holdout = holdout_data[target_col]
        h_preds = best_pipe.predict(X_holdout)

        if is_classification:
            h_probs = best_pipe.predict_proba(X_holdout)[:, 1] if hasattr(best_pipe, "predict_proba") else None
            holdout_results = evaluate_classification(y_holdout.to_numpy(), h_preds, h_probs)
            print(f"Holdout ROC-AUC: {holdout_results['roc_auc']:.4f} | Accuracy: {holdout_results['accuracy']:.2%} | Bal Acc: {holdout_results['balanced_accuracy']:.2%}")
        else:
            dev_mean = float(dev_data[target_col].mean())
            holdout_data["_pred"] = h_preds
            holdout_results = evaluate_regression(y_holdout.to_numpy(), h_preds, dev_mean, holdout_data, "_pred", target_col)
            print(f"Holdout MAE: {holdout_results['mae']:.4f} | RMSE: {holdout_results['rmse']:.4f} | OOS R²: {holdout_results['r2_vs_train_mean']:.4f} | Rank IC: {holdout_results['mean_rank_ic']:.4f}")
    else:
        print("\n[NOTE] Holdout set remains strictly UNTOUCHED as requested. Set --evaluate-holdout to evaluate.")

    return {
        "task": task,
        "feature_subset": feature_subset,
        "best_model_name": best_model_name,
        "walk_forward_df": wf_df,
        "best_pipeline": best_pipe,
        "holdout_results": holdout_results,
    }


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


def main():
    parser = argparse.ArgumentParser(description="NSE Stock Prediction Quantitative Training Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast synthetic smoke test")
    parser.add_argument("--run-training", action="store_true", help="Execute the formal quantitative training pipeline")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "regression"],
                        help="Prediction task: 'classification' (5d direction) or 'regression' (63d return)")
    parser.add_argument("--feature-subset", type=str, default="market", choices=["market", "fundamentals", "combined"],
                        help="Feature configuration: 'market', 'fundamentals', or 'combined'")
    parser.add_argument("--evaluate-holdout", action="store_true",
                        help="Evaluate on the final sequestered holdout partition (default: False)")
    args = parser.parse_args()

    if args.run_training:
        run_training_experiment(
            task=args.task,
            feature_subset=args.feature_subset,
            evaluate_holdout=args.evaluate_holdout,
        )
    else:
        run_synthetic_smoke_test()


if __name__ == "__main__":
    main()
