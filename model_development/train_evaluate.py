"""
Training and Chronological Walk-Forward Validation Module for NSE Stock Prediction.

Implements:
1. Strict chronological date partitioning with purging/embargo.
2. Expanding-window walk-forward validation across multiple folds.
3. Untouched final chronological holdout evaluation.
4. Comprehensive multi-metric evaluation (ROC-AUC, PR-AUC, Accuracy, F1, Brier, Calibration).
5. Evaluation of diverse models:
   - Baseline (Most Frequent)
   - Logistic Regression (L2 regularized, balanced)
   - Random Forest Classifier
   - HistGradientBoosting Classifier
   - XGBoost Classifier
   - Future Return Regression (HistGradientBoosting Regressor secondary experiment)
6. Feature importance analysis and per-stock holdout performance breakdown.
7. Diagnostic visualizations (ROC/PR curves, confusion matrix, calibration, walk-forward trends).
8. Pipeline persistence and predictions export.
"""

from pathlib import Path
import json
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
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
    classification_report,
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
    HORIZON_SESSIONS,
    PURGE_SESSIONS,
    RANDOM_STATE,
    HOLDOUT_FRACTION,
)

warnings.filterwarnings("ignore")


def build_model_pipelines() -> dict:
    """
    Constructs isolated scikit-learn Pipelines for each model candidate.
    Every pipeline encapsulates its own imputer and scaler to prevent data leakage.
    """
    models = {
        "Dummy_Baseline": Pipeline([
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
            # Native support for missing values, but median imputer ensures uniform inputs
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
    return models


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculates comprehensive classification evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # ROC-AUC and PR-AUC require probabilities
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(np.mean(y_true))
        metrics["brier_score"] = 0.25

    return metrics


def run_walk_forward_validation(dev_data: pd.DataFrame, dates: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs expanding-window walk-forward validation across chronological folds.
    Applies a 5-session purge window between training and validation in every fold.
    """
    print("\n--- Running Chronological Walk-Forward Validation ---")
    n_dates = len(dates)
    fold_specs = [
        (0.40, 0.60),
        (0.60, 0.80),
        (0.80, 1.00),
    ]

    records = []

    for fold_idx, (train_pct, val_pct) in enumerate(fold_specs, 1):
        train_end_idx = int(n_dates * train_pct)
        val_start_idx = train_end_idx + PURGE_SESSIONS
        val_end_idx = int(n_dates * val_pct)

        train_dates = dates[:train_end_idx]
        val_dates = dates[val_start_idx:val_end_idx]

        train_fold = dev_data[dev_data["Date"].isin(train_dates)]
        val_fold = dev_data[dev_data["Date"].isin(val_dates)]

        X_train = train_fold[FEATURE_COLUMNS]
        y_train = train_fold["target_up_5d"].astype(int)
        X_val = val_fold[FEATURE_COLUMNS]
        y_val = val_fold["target_up_5d"].astype(int)

        print(f"Fold {fold_idx}: Train {len(train_fold):,} rows ({train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}) "
              f"-> Purged {PURGE_SESSIONS} days -> Val {len(val_fold):,} rows ({val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')})")

        models = build_model_pipelines()
        for name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_val)[:, 1]
            else:
                probs = np.full(len(X_val), 0.5)

            metrics = evaluate_predictions(y_val.to_numpy(), preds, probs)
            metrics["fold"] = fold_idx
            metrics["model"] = name
            metrics["train_rows"] = len(train_fold)
            metrics["val_rows"] = len(val_fold)
            records.append(metrics)

    results_df = pd.DataFrame(records)
    results_df.to_csv(WALK_FORWARD_RESULTS_PATH, index=False)

    # Aggregated summary
    summary_df = results_df.groupby("model")[["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc", "brier_score"]].agg(["mean", "std"])
    summary_df.to_csv(WALK_FORWARD_SUMMARY_PATH)

    print("\nWalk-Forward Validation Summary (Mean +/- Std):")
    for model_name, grp in results_df.groupby("model"):
        print(f"  {model_name:25s} | ROC-AUC: {grp['roc_auc'].mean():.4f} +/- {grp['roc_auc'].std():.4f} "
              f"| Bal Acc: {grp['balanced_accuracy'].mean():.4f} | Acc: {grp['accuracy'].mean():.4f}")

    return results_df, summary_df


def run_holdout_evaluation(train_data: pd.DataFrame, holdout_data: pd.DataFrame, best_model_name: str) -> dict:
    """
    Trains all models on the development set and evaluates on the untouched holdout set.
    Persists holdout predictions and evaluation artifacts.
    """
    print("\n--- Running Final Chronological Holdout Evaluation ---")
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["target_up_5d"].astype(int)
    X_holdout = holdout_data[FEATURE_COLUMNS]
    y_holdout = holdout_data["target_up_5d"].astype(int)

    print(f"Holdout Set: {len(holdout_data):,} rows from {holdout_data['Date'].min().strftime('%Y-%m-%d')} to {holdout_data['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Holdout Class Distribution: Class 1 = {(y_holdout == 1).mean():.1%}, Class 0 = {(y_holdout == 0).mean():.1%}")

    models = build_model_pipelines()
    holdout_records = []
    predictions_dict = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_holdout)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_holdout)[:, 1]
        else:
            probs = np.full(len(X_holdout), 0.5)

        predictions_dict[f"{name}_pred"] = preds
        predictions_dict[f"{name}_prob"] = probs

        metrics = evaluate_predictions(y_holdout.to_numpy(), preds, probs)
        metrics["model"] = name
        metrics["train_rows"] = len(train_data)
        metrics["holdout_rows"] = len(holdout_data)
        holdout_records.append(metrics)

    comparison_df = pd.DataFrame(holdout_records)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    print("\nHoldout Model Comparison:")
    print(comparison_df[["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

    # Save holdout predictions table
    pred_export_df = holdout_data[["Symbol", "Date", "adj_close", "future_adj_close_5d", "future_return_5d", "target_up_5d"]].copy()
    for col, vals in predictions_dict.items():
        pred_export_df[col] = vals
    pred_export_df.to_csv(HOLDOUT_PREDICTIONS_PATH, index=False)

    # Secondary experiment: Return regression
    print("\n--- Secondary Experiment: Continuous 5-Day Return Regression ---")
    reg_train = train_data.dropna(subset=["future_return_5d"]).copy()
    reg_holdout = holdout_data.dropna(subset=["future_return_5d"]).copy()

    reg_pipeline = Pipeline([
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
    ])
    reg_pipeline.fit(reg_train[FEATURE_COLUMNS], reg_train["future_return_5d"])
    reg_preds = reg_pipeline.predict(reg_holdout[FEATURE_COLUMNS])

    reg_mae = float(mean_absolute_error(reg_holdout["future_return_5d"], reg_preds))
    reg_rmse = float(root_mean_squared_error(reg_holdout["future_return_5d"], reg_preds))
    # Directional accuracy: sign(predicted return) == sign(actual return)
    actual_direction = (reg_holdout["future_return_5d"] > 0).astype(int)
    predicted_direction = (reg_preds > 0).astype(int)
    reg_dir_acc = float(accuracy_score(actual_direction, predicted_direction))
    reg_dir_auc = float(roc_auc_score(actual_direction, reg_preds))

    print(f"Regression MAE: {reg_mae:.4f} | RMSE: {reg_rmse:.4f}")
    print(f"Regression Directional Accuracy: {reg_dir_acc:.4f} | Rank/ROC-AUC: {reg_dir_auc:.4f}")

    # Save best selected model
    best_pipeline = models[best_model_name]
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"\nPersisted best model ({best_model_name}) pipeline to: {BEST_MODEL_PATH}")

    # Per-stock breakdown on holdout for best model
    per_stock_records = []
    best_probs = predictions_dict[f"{best_model_name}_prob"]
    best_preds = predictions_dict[f"{best_model_name}_pred"]

    for symbol in sorted(holdout_data["Symbol"].unique()):
        mask = (holdout_data["Symbol"] == symbol).to_numpy()
        sym_y = y_holdout.to_numpy()[mask]
        sym_p = best_probs[mask]
        sym_pred = best_preds[mask]

        if len(np.unique(sym_y)) > 1:
            sym_auc = float(roc_auc_score(sym_y, sym_p))
        else:
            sym_auc = np.nan

        per_stock_records.append({
            "symbol": symbol,
            "observations": int(mask.sum()),
            "class_1_ratio": float(np.mean(sym_y)),
            "accuracy": float(accuracy_score(sym_y, sym_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(sym_y, sym_pred)),
            "f1": float(f1_score(sym_y, sym_pred, zero_division=0)),
            "roc_auc": sym_auc,
        })

    per_stock_df = pd.DataFrame(per_stock_records)
    per_stock_df.to_csv(PER_STOCK_EVAL_PATH, index=False)

    return {
        "comparison_df": comparison_df,
        "predictions_dict": predictions_dict,
        "regression_results": {
            "mae": reg_mae,
            "rmse": reg_rmse,
            "directional_accuracy": reg_dir_acc,
            "directional_auc": reg_dir_auc,
        },
        "per_stock_df": per_stock_df,
        "best_pipeline": best_pipeline,
    }


def generate_visualizations(
    walk_forward_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    predictions_dict: dict,
    best_model_name: str,
    feature_names: list,
    best_pipeline: Pipeline,
    per_stock_df: pd.DataFrame
):
    """Generates all publication-quality diagnostic charts and saves them to disk."""
    print("\n--- Generating Evaluation Charts and Figures ---")
    sns.set_theme(style="whitegrid", palette="muted")
    y_holdout = holdout_df["target_up_5d"].astype(int).to_numpy()

    # 1. Walk-Forward Fold Performance
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=walk_forward_df,
        x="fold",
        y="roc_auc",
        hue="model",
    )
    plt.axhline(0.50, color="red", linestyle="--", alpha=0.7, label="Chance Baseline (0.50)")
    plt.title("Walk-Forward ROC-AUC Across Chronological Expanding Folds", fontsize=13, fontweight="bold")
    plt.xlabel("Walk-Forward Fold Number")
    plt.ylabel("Out-of-Sample ROC-AUC")
    plt.ylim(0.40, 0.65)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "walk_forward_auc.png", dpi=300)
    plt.close()

    # 2. Holdout ROC and PR Curves
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name in ["Dummy_Baseline", "Logistic_Regression", "Random_Forest", "Hist_Gradient_Boosting", "XGBoost"]:
        probs = predictions_dict[f"{name}_prob"]
        fpr, tpr, _ = roc_curve(y_holdout, probs)
        auc_val = roc_auc_score(y_holdout, probs)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", lw=1.8)

        prec, rec, _ = precision_recall_curve(y_holdout, probs)
        ap_val = average_precision_score(y_holdout, probs)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap_val:.3f})", lw=1.8)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("Holdout ROC Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].axhline(np.mean(y_holdout), color="k", linestyle="--", alpha=0.5, label="No Skill")
    axes[1].set_title("Holdout Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "holdout_roc_pr_curves.png", dpi=300)
    plt.close()

    # 3. Holdout Confusion Matrix Heatmap (Best Model)
    best_preds = predictions_dict[f"{best_model_name}_pred"]
    cm = confusion_matrix(y_holdout, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Down (0)", "Pred Up (1)"],
                yticklabels=["Actual Down (0)", "Actual Up (1)"])
    plt.title(f"Holdout Confusion Matrix: {best_model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "holdout_confusion_matrix.png", dpi=300)
    plt.close()

    # 4. Probability Calibration Curve (Reliability Diagram)
    plt.figure(figsize=(7, 6))
    for name in ["Logistic_Regression", "Random_Forest", "Hist_Gradient_Boosting", "XGBoost"]:
        probs = predictions_dict[f"{name}_prob"]
        prob_true, prob_pred = calibration_curve(y_holdout, probs, n_bins=8, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", label=name, lw=1.8)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfectly Calibrated")
    plt.title("Holdout Probability Calibration (Reliability Diagram)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probability_calibration.png", dpi=300)
    plt.close()

    # 5. Feature Importance Analysis
    model_obj = best_pipeline.named_steps["model"]
    importances = None
    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
    elif hasattr(model_obj, "coef_"):
        importances = np.abs(model_obj.coef_[0])

    if importances is not None:
        feat_imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=True)

        plt.figure(figsize=(8, 8))
        plt.barh(feat_imp_df["feature"], feat_imp_df["importance"], color="#3470a3")
        plt.title(f"Feature Importances ({best_model_name})", fontsize=13, fontweight="bold")
        plt.xlabel("Relative Importance Metric")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300)
        plt.close()

    # 6. Per-Stock ROC-AUC Bar Chart on Holdout
    plt.figure(figsize=(10, 5))
    valid_stocks = per_stock_df.dropna(subset=["roc_auc"]).sort_values("roc_auc")
    plt.barh(valid_stocks["symbol"], valid_stocks["roc_auc"], color="#2b8cbe")
    plt.axvline(0.50, color="red", linestyle="--", alpha=0.7, label="Chance (0.50)")
    plt.title(f"Holdout ROC-AUC per Symbol ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Holdout ROC-AUC")
    plt.xlim(0.35, 0.70)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_stock_auc.png", dpi=300)
    plt.close()

    print(f"All diagnostic figures successfully saved to: {FIGURES_DIR}")


def execute_training_workflow():
    """Master workflow for cross-validation, holdout evaluation, and reporting."""
    print("=" * 70)
    print("EXECUTING MODEL TRAINING & CHRONOLOGICAL EVALUATION")
    print("=" * 70)

    # 1. Load analytical dataset
    if not ANALYTICAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Analytical dataset not found at: {ANALYTICAL_DATASET_PATH}. Run data_pipeline.py first.")

    df = pd.read_csv(ANALYTICAL_DATASET_PATH, parse_dates=["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # Eligible modeling rows must have known target label
    labeled_df = df.dropna(subset=["target_up_5d"]).copy()
    labeled_df["target_up_5d"] = labeled_df["target_up_5d"].astype(int)

    dates = np.array(sorted(labeled_df["Date"].unique()))
    n_dates = len(dates)

    # 2. Chronological Train-Dev vs Holdout Split
    holdout_split_idx = int(n_dates * (1.0 - HOLDOUT_FRACTION))
    holdout_start_idx = holdout_split_idx + PURGE_SESSIONS

    dev_dates = dates[:holdout_split_idx]
    holdout_dates = dates[holdout_start_idx:]

    dev_data = labeled_df[labeled_df["Date"].isin(dev_dates)].copy()
    holdout_data = labeled_df[labeled_df["Date"].isin(holdout_dates)].copy()

    print(f"Total labeled observations: {len(labeled_df):,} across {n_dates} trading dates.")
    print(f"Development Set: {len(dev_data):,} rows ({dev_dates[0].strftime('%Y-%m-%d')} to {dev_dates[-1].strftime('%Y-%m-%d')})")
    print(f"Purge Window: {PURGE_SESSIONS} trading dates between dev and holdout.")
    print(f"Holdout Set: {len(holdout_data):,} rows ({holdout_dates[0].strftime('%Y-%m-%d')} to {holdout_dates[-1].strftime('%Y-%m-%d')})")

    # 3. Walk-Forward Cross-Validation on Development Set
    wf_results, wf_summary = run_walk_forward_validation(dev_data, dev_dates)

    # 4. Model Selection based on Walk-Forward Mean ROC-AUC
    mean_auc_per_model = wf_results.groupby("model")["roc_auc"].mean()
    # Exclude dummy baseline from selection
    candidate_models = mean_auc_per_model.drop("Dummy_Baseline", errors="ignore")
    best_model_name = candidate_models.idxmax()
    print(f"\nSelected Best Model based on Walk-Forward ROC-AUC: {best_model_name} (Mean WF AUC: {candidate_models[best_model_name]:.4f})")

    # 5. Untouched Holdout Evaluation
    holdout_output = run_holdout_evaluation(dev_data, holdout_data, best_model_name)

    # 6. Generate Figures
    generate_visualizations(
        walk_forward_df=wf_results,
        holdout_df=holdout_data,
        predictions_dict=holdout_output["predictions_dict"],
        best_model_name=best_model_name,
        feature_names=FEATURE_COLUMNS,
        best_pipeline=holdout_output["best_pipeline"],
        per_stock_df=holdout_output["per_stock_df"],
    )

    print("=" * 70)
    print("MODEL TRAINING & CHRONOLOGICAL EVALUATION COMPLETE")
    print("=" * 70)

    return {
        "best_model_name": best_model_name,
        "wf_summary": wf_summary,
        "holdout_comparison": holdout_output["comparison_df"],
        "regression_results": holdout_output["regression_results"],
        "per_stock": holdout_output["per_stock_df"],
    }


if __name__ == "__main__":
    execute_training_workflow()
