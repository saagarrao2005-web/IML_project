"""
Official One-Time Holdout Evaluation Script for NSE Stock Prediction.

Strict Research Protocol Enforcement:
1. Loads already-frozen candidate models from outputs/models/.
2. Evaluates on the strictly sequestered holdout partition:
   - Task A: 2,896 observations (2026-01-22 to 2026-10-15) across 16 symbols.
   - Task B: 1,824 observations (2026-02-04 to 2026-07-23) across 16 symbols.
3. Preprocessing and model fitting are strictly locked to the development set.
4. Zero holdout information is used for fitting, calibration, thresholding, or selection.
5. Saves all final artifacts under outputs/final_holdout/.
"""

import sys
from pathlib import Path

# Add model_development to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import logging
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
from sklearn.dummy import DummyClassifier

from config import (
    ANALYTICAL_DATASET_PATH,
    OUTPUTS_DIR,
    MODELS_DIR,
    MARKET_FEATURES,
    EXPECTED_SYMBOLS,
    RANDOM_STATE,
    HOLDOUT_FRACTION,
    PURGE_SESSIONS,
    PURGE_REGRESSION_SESSIONS,
)
from training_pipeline import (
    build_classification_models,
    build_regression_models,
    evaluate_classification,
    evaluate_regression,
    compute_cross_sectional_rank_ic,
)


def setup_holdout_dirs() -> dict:
    base = OUTPUTS_DIR / "final_holdout"
    preds_dir = base / "predictions"
    reports_dir = base / "reports"
    models_dir = base / "models"
    logs_dir = base / "logs"

    for d in [preds_dir, reports_dir, models_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "predictions": preds_dir,
        "reports": reports_dir,
        "models": models_dir,
        "logs": logs_dir,
    }


def run_holdout_evaluation():
    dirs = setup_holdout_dirs()
    log_path = dirs["logs"] / "holdout_evaluation.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("HoldoutEvaluator")

    logger.info("=" * 80)
    logger.info("EXECUTING ONE-TIME FINAL HOLDOUT EVALUATION (NSE STOCK PREDICTION)")
    logger.info("=" * 80)

    # 1. Load analytical dataset
    if not ANALYTICAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Analytical dataset not found at {ANALYTICAL_DATASET_PATH}")
    df = pd.read_csv(ANALYTICAL_DATASET_PATH, parse_dates=["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    logger.info(f"Loaded analytical dataset: {len(df):,} total rows across {df['Symbol'].nunique()} symbols.")

    # 2. Check frozen candidate models on disk
    frozen_cls_path = MODELS_DIR / "classification_best_dev_model.joblib"
    frozen_reg_path = MODELS_DIR / "regression_best_dev_model.joblib"

    if not frozen_cls_path.exists():
        raise FileNotFoundError(f"CRITICAL: Frozen classification candidate artifact missing at {frozen_cls_path}")
    if not frozen_reg_path.exists():
        raise FileNotFoundError(f"CRITICAL: Frozen regression candidate artifact missing at {frozen_reg_path}")

    logger.info(f"Found frozen classification candidate: {frozen_cls_path}")
    logger.info(f"Found frozen regression candidate:     {frozen_reg_path}")

    # ==========================================================================
    # TASK A: 5-DAY DIRECTIONAL CLASSIFICATION HOLDOUT
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TASK A: FINAL HOLDOUT EVALUATION (5-DAY DIRECTIONAL CLASSIFICATION)")
    logger.info("=" * 80)

    target_col_a = "target_up_5d"
    labeled_a = df.dropna(subset=[target_col_a]).copy()
    labeled_a[target_col_a] = labeled_a[target_col_a].astype(int)

    dates_a = np.array(sorted(labeled_a["Date"].unique()))
    n_dates_a = len(dates_a)
    holdout_split_a = int(n_dates_a * (1.0 - HOLDOUT_FRACTION))
    holdout_start_a = holdout_split_a + PURGE_SESSIONS

    dev_dates_a = dates_a[:holdout_split_a]
    holdout_dates_a = dates_a[holdout_start_a:]

    dev_data_a = labeled_a[labeled_a["Date"].isin(dev_dates_a)].copy()
    holdout_data_a = labeled_a[labeled_a["Date"].isin(holdout_dates_a)].copy()

    logger.info(f"Task A Development rows: {len(dev_data_a):,} ({dev_dates_a[0].strftime('%Y-%m-%d')} to {dev_dates_a[-1].strftime('%Y-%m-%d')})")
    logger.info(f"Task A Purge window:      {PURGE_SESSIONS} trading sessions")
    logger.info(f"Task A Holdout rows:     {len(holdout_data_a):,} ({holdout_dates_a[0].strftime('%Y-%m-%d')} to {holdout_dates_a[-1].strftime('%Y-%m-%d')})")
    logger.info(f"Task A Holdout class balance: Class 1 = {(holdout_data_a[target_col_a] == 1).sum():,} ({(holdout_data_a[target_col_a] == 1).mean():.2%}), Class 0 = {(holdout_data_a[target_col_a] == 0).sum():,}")

    assert len(holdout_data_a) == 2896, f"Expected 2896 holdout rows, got {len(holdout_data_a)}"
    assert holdout_data_a["Symbol"].nunique() == 16, "Expected 16 symbols"

    # Load frozen candidate
    cls_candidate = joblib.load(frozen_cls_path)
    logger.info(f"Loaded frozen Task A candidate: {type(cls_candidate.named_steps['model']).__name__}")

    # Build and fit full development models for prespecified comparators
    # Strictly fitted on dev_data_a, zero holdout data used
    cls_models = build_classification_models()
    cls_preds_dict = {
        "Date": holdout_data_a["Date"].dt.strftime("%Y-%m-%d").values,
        "Symbol": holdout_data_a["Symbol"].values,
        "actual": holdout_data_a[target_col_a].values,
    }

    cls_summary_records = []
    y_holdout_a = holdout_data_a[target_col_a].to_numpy()

    for m_name, pipe in cls_models.items():
        if m_name == "Logistic_Regression":
            # Use the exact already-frozen artifact from disk
            pipe = cls_candidate
        else:
            # Fit prespecified comparator on development data
            pipe.fit(dev_data_a[MARKET_FEATURES], dev_data_a[target_col_a])
            # Save comparator model artifact
            joblib.dump(pipe, dirs["models"] / f"classification_{m_name}_dev_model.joblib")

        preds = pipe.predict(holdout_data_a[MARKET_FEATURES])
        probs = pipe.predict_proba(holdout_data_a[MARKET_FEATURES])[:, 1] if hasattr(pipe, "predict_proba") else None

        cls_preds_dict[f"{m_name}_pred"] = preds
        if probs is not None:
            cls_preds_dict[f"{m_name}_prob"] = probs

        m_metrics = evaluate_classification(y_holdout_a, preds, probs)
        m_metrics["model"] = m_name
        m_metrics["sample_size"] = len(holdout_data_a)
        m_metrics["class_1_count"] = int(np.sum(y_holdout_a == 1))
        m_metrics["class_0_count"] = int(np.sum(y_holdout_a == 0))
        m_metrics["candidate_status"] = "CHAMPION" if m_name == "Logistic_Regression" else "COMPARATOR"
        cls_summary_records.append(m_metrics)

    cls_summary_df = pd.DataFrame(cls_summary_records)
    cls_preds_df = pd.DataFrame(cls_preds_dict)

    # Save Task A predictions and summary
    cls_preds_path = dirs["predictions"] / "holdout_classification_predictions.csv"
    cls_preds_df.to_csv(cls_preds_path, index=False)
    logger.info(f"Saved Task A holdout predictions to: {cls_preds_path}")

    cls_summary_path = dirs["reports"] / "holdout_classification_summary.csv"
    cls_summary_to_save = cls_summary_df.drop(columns=["confusion_matrix"], errors="ignore")
    cls_summary_to_save.to_csv(cls_summary_path, index=False)
    logger.info(f"Saved Task A holdout summary to: {cls_summary_path}")

    # Per-stock evaluation for candidate model (Logistic Regression)
    per_stock_cls = []
    log_reg_preds = cls_preds_dict["Logistic_Regression_pred"]
    log_reg_probs = cls_preds_dict["Logistic_Regression_prob"]

    for sym in sorted(holdout_data_a["Symbol"].unique()):
        mask = (holdout_data_a["Symbol"] == sym).values
        sym_true = y_holdout_a[mask]
        sym_pred = log_reg_preds[mask]
        sym_prob = log_reg_probs[mask]

        sym_metrics = evaluate_classification(sym_true, sym_pred, sym_prob)
        sym_metrics["symbol"] = sym
        sym_metrics["sample_size"] = len(sym_true)
        sym_metrics["class_1_ratio"] = float(np.mean(sym_true))
        per_stock_cls.append(sym_metrics)

    per_stock_cls_df = pd.DataFrame(per_stock_cls)
    per_stock_cls_path = dirs["reports"] / "holdout_per_stock_classification.csv"
    per_stock_cls_df.drop(columns=["confusion_matrix"], errors="ignore").to_csv(per_stock_cls_path, index=False)
    logger.info(f"Saved Task A per-stock classification to: {per_stock_cls_path}")

    # Print Task A holdout comparison table
    logger.info("\nTask A Holdout Evaluation Summary:")
    header = f"{'Model':28s} | {'ROC-AUC':9s} | {'Bal Acc':9s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'F1':9s} | {'Brier':9s}"
    logger.info(header)
    logger.info("-" * len(header))
    for _, r in cls_summary_df.iterrows():
        logger.info(f"{r['model']:28s} | {r['roc_auc']:.4f}    | {r['balanced_accuracy']:.4f}    | {r['accuracy']:.4f}    | {r['precision']:.4f}    | {r['recall']:.4f}    | {r['f1']:.4f}   | {r['brier_score']:.4f}")

    # ==========================================================================
    # TASK B: 63-DAY CONTINUOUS-RETURN REGRESSION HOLDOUT
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TASK B: FINAL HOLDOUT EVALUATION (63-DAY CONTINUOUS-RETURN REGRESSION)")
    logger.info("=" * 80)

    target_col_b = "future_return_63d"
    labeled_b = df.dropna(subset=[target_col_b]).copy()

    dates_b = np.array(sorted(labeled_b["Date"].unique()))
    n_dates_b = len(dates_b)
    holdout_split_b = int(n_dates_b * (1.0 - HOLDOUT_FRACTION))
    holdout_start_b = holdout_split_b + PURGE_REGRESSION_SESSIONS

    dev_dates_b = dates_b[:holdout_split_b]
    holdout_dates_b = dates_b[holdout_start_b:]

    dev_data_b = labeled_b[labeled_b["Date"].isin(dev_dates_b)].copy()
    holdout_data_b = labeled_b[labeled_b["Date"].isin(holdout_dates_b)].copy()

    dev_mean_b = float(dev_data_b[target_col_b].mean())

    logger.info(f"Task B Development rows: {len(dev_data_b):,} ({dev_dates_b[0].strftime('%Y-%m-%d')} to {dev_dates_b[-1].strftime('%Y-%m-%d')})")
    logger.info(f"Task B Purge window:      {PURGE_REGRESSION_SESSIONS} trading sessions (~3 calendar months)")
    logger.info(f"Task B Holdout rows:     {len(holdout_data_b):,} ({holdout_dates_b[0].strftime('%Y-%m-%d')} to {holdout_dates_b[-1].strftime('%Y-%m-%d')})")
    logger.info(f"Task B Development Mean: {dev_mean_b:.6f} (+{dev_mean_b*100:.2f}%)")
    logger.info(f"Task B Holdout Mean:     {float(holdout_data_b[target_col_b].mean()):.6f} (+{float(holdout_data_b[target_col_b].mean())*100:.2f}%)")

    assert len(holdout_data_b) == 1824, f"Expected 1824 holdout rows, got {len(holdout_data_b)}"
    assert holdout_data_b["Symbol"].nunique() == 16, "Expected 16 symbols"

    # Load frozen regression candidate
    reg_candidate = joblib.load(frozen_reg_path)
    logger.info(f"Loaded frozen Task B candidate: {type(reg_candidate.named_steps['model']).__name__}")

    # Build prespecified regression models
    reg_models = build_regression_models()
    reg_preds_dict = {
        "Date": holdout_data_b["Date"].dt.strftime("%Y-%m-%d").values,
        "Symbol": holdout_data_b["Symbol"].values,
        "actual": holdout_data_b[target_col_b].values,
    }

    reg_summary_records = []
    y_holdout_b = holdout_data_b[target_col_b].to_numpy()

    for m_name, pipe in reg_models.items():
        if m_name == "Random_Forest_Regressor":
            # Use exact frozen candidate artifact from disk
            pipe = reg_candidate
        else:
            # Fit prespecified baseline or comparator strictly on development data
            pipe.fit(dev_data_b[MARKET_FEATURES], dev_data_b[target_col_b])
            joblib.dump(pipe, dirs["models"] / f"regression_{m_name}_dev_model.joblib")

        preds = pipe.predict(holdout_data_b[MARKET_FEATURES])
        reg_preds_dict[f"{m_name}_pred"] = preds

        holdout_data_b["_pred"] = preds
        m_metrics = evaluate_regression(y_holdout_b, preds, dev_mean_b, holdout_data_b, "_pred", target_col_b)
        m_metrics["model"] = m_name
        m_metrics["sample_size"] = len(holdout_data_b)
        m_metrics["candidate_status"] = "CHAMPION" if m_name == "Random_Forest_Regressor" else ("BENCHMARK" if "Baseline" in m_name else "COMPARATOR")
        reg_summary_records.append(m_metrics)

    reg_summary_df = pd.DataFrame(reg_summary_records)
    reg_preds_df = pd.DataFrame(reg_preds_dict)

    # Save Task B predictions and summary
    reg_preds_path = dirs["predictions"] / "holdout_regression_predictions.csv"
    reg_preds_df.to_csv(reg_preds_path, index=False)
    logger.info(f"Saved Task B holdout predictions to: {reg_preds_path}")

    reg_summary_path = dirs["reports"] / "holdout_regression_summary.csv"
    reg_summary_df.to_csv(reg_summary_path, index=False)
    logger.info(f"Saved Task B holdout summary to: {reg_summary_path}")

    # Per-stock evaluation for candidate model (Random Forest Regressor) and Zero Return Baseline
    per_stock_reg = []
    rf_preds = reg_preds_dict["Random_Forest_Regressor_pred"]
    zero_preds = reg_preds_dict["Zero_Return_Baseline_pred"]

    for sym in sorted(holdout_data_b["Symbol"].unique()):
        mask = (holdout_data_b["Symbol"] == sym).values
        sym_sub = holdout_data_b[mask].copy()
        sym_true = y_holdout_b[mask]
        sym_pred = rf_preds[mask]

        sym_sub["_rf_pred"] = sym_pred
        sym_m = evaluate_regression(sym_true, sym_pred, dev_mean_b, sym_sub, "_rf_pred", target_col_b)
        sym_m["symbol"] = sym
        sym_m["sample_size"] = len(sym_true)
        sym_m["actual_mean_return"] = float(np.mean(sym_true))
        sym_m["zero_baseline_mae"] = float(mean_absolute_error(sym_true, zero_preds[mask]))
        per_stock_reg.append(sym_m)

    per_stock_reg_df = pd.DataFrame(per_stock_reg)
    per_stock_reg_path = dirs["reports"] / "holdout_per_stock_regression.csv"
    per_stock_reg_df.to_csv(per_stock_reg_path, index=False)
    logger.info(f"Saved Task B per-stock regression to: {per_stock_reg_path}")

    # Print Task B holdout comparison table
    logger.info("\nTask B Holdout Evaluation Summary:")
    header_b = f"{'Model':32s} | {'MAE':9s} | {'RMSE':9s} | {'OOS R²(Mean)':12s} | {'OOS R²(Zero)':12s} | {'Dir Acc':9s} | {'Rank IC':9s} | {'t-stat':7s}"
    logger.info(header_b)
    logger.info("-" * len(header_b))
    for _, r in reg_summary_df.iterrows():
        logger.info(f"{r['model']:32s} | {r['mae']:.4f}    | {r['rmse']:.4f}    | {r['r2_vs_train_mean']:12.4f} | {r['r2_vs_zero']:12.4f} | {r['directional_accuracy']:.4f}    | {r['mean_rank_ic']:.4f}    | {r['rank_ic_t_stat']:6.2f}")

    # ==========================================================================
    # METADATA & REPRODUCIBILITY PROVENANCE EXPORT
    # ==========================================================================
    holdout_meta = {
        "execution_timestamp": pd.Timestamp.now().isoformat(),
        "holdout_evaluated_status": True,
        "task_a": {
            "target": target_col_a,
            "horizon_sessions": 5,
            "purge_sessions": PURGE_SESSIONS,
            "n_holdout_dates": len(holdout_dates_a),
            "holdout_date_range": [holdout_dates_a[0].strftime("%Y-%m-%d"), holdout_dates_a[-1].strftime("%Y-%m-%d")],
            "n_holdout_rows": len(holdout_data_a),
            "candidate_model": "Logistic_Regression",
            "candidate_artifact": str(frozen_cls_path),
            "summary_metrics": cls_summary_df.to_dict(orient="records"),
        },
        "task_b": {
            "target": target_col_b,
            "horizon_sessions": 63,
            "purge_sessions": PURGE_REGRESSION_SESSIONS,
            "n_holdout_dates": len(holdout_dates_b),
            "holdout_date_range": [holdout_dates_b[0].strftime("%Y-%m-%d"), holdout_dates_b[-1].strftime("%Y-%m-%d")],
            "n_holdout_rows": len(holdout_data_b),
            "dev_mean": dev_mean_b,
            "candidate_model": "Random_Forest_Regressor",
            "candidate_artifact": str(frozen_reg_path),
            "summary_metrics": reg_summary_df.to_dict(orient="records"),
        },
        "universe_symbols": EXPECTED_SYMBOLS,
        "n_symbols": len(EXPECTED_SYMBOLS),
        "random_state": RANDOM_STATE,
    }

    meta_path = dirs["reports"] / "final_holdout_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(holdout_meta, f, indent=2, default=str)
    logger.info(f"\nSaved final holdout metadata JSON to: {meta_path}")
    logger.info("\nFinal holdout evaluation completed successfully.")


if __name__ == "__main__":
    run_holdout_evaluation()
