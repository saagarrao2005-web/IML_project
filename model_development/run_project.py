"""
Master Execution Pipeline for the NSE Stock Price Prediction ML Project.

Runs the entire autonomous quantitative research and engineering workflow:
1. Environment and configuration verification.
2. Ingestion, corporate action backward-adjustment, feature engineering, and target construction.
3. Rigorous data quality and target alignment auditing.
4. Chronological walk-forward cross-validation across multiple folds with purging.
5. Untouched chronological holdout evaluation and baseline comparison.
6. Diagnostic visualization generation (ROC/PR curves, confusion matrix, calibration, per-stock).
7. Execution of the automated test suite via pytest.
8. Generation of final reports, model artifacts, and predictions.
"""

import sys
import subprocess
import time
from pathlib import Path
import pandas as pd

# Add model_development to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    ANALYTICAL_DATASET_PATH,
    BEST_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    WALK_FORWARD_SUMMARY_PATH,
    CORPORATE_ACTION_REGISTER_PATH,
    DATA_QUALITY_REPORT_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    EXPECTED_SYMBOLS,
)
import data_pipeline
import train_evaluate


def print_banner(text: str):
    print("\n" + "=" * 76)
    print(f"  {text}")
    print("=" * 76 + "\n")


def run_tests() -> bool:
    """Executes the test suite via pytest subprocess."""
    print_banner("STEP 3: RUNNING AUTOMATED TEST SUITE")
    test_path = BASE_DIR / "tests"
    cmd = [sys.executable, "-m", "pytest", str(test_path), "-v"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        return False
    return True


def run_full_pipeline():
    start_time = time.time()
    print_banner("STARTING FULL END-TO-END NSE STOCK PREDICTION PIPELINE")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project Directory: {BASE_DIR}")
    print(f"Target Universe: {len(EXPECTED_SYMBOLS)} NSE Equities: {', '.join(EXPECTED_SYMBOLS)}")

    # 1. Data Pipeline
    print_banner("STEP 1: INGESTION, AUDITING, ADJUSTMENT & FEATURE ENGINEERING")
    df_analytical = data_pipeline.run_pipeline()

    # 2. Modeling & Validation
    print_banner("STEP 2: MODEL TRAINING, WALK-FORWARD VALIDATION & HOLDOUT EVALUATION")
    eval_results = train_evaluate.execute_training_workflow()

    # 3. Automated Tests
    tests_passed = run_tests()
    if not tests_passed:
        print("WARNING: Some automated tests failed! Review test outputs above.")
    else:
        print("SUCCESS: All automated pipeline tests passed successfully.")

    # 4. Final Executive Summary
    elapsed = time.time() - start_time
    print_banner("PIPELINE EXECUTION COMPLETE - EXECUTIVE SUMMARY")

    wf_summary = pd.read_csv(WALK_FORWARD_SUMMARY_PATH, header=[0, 1], index_col=0)
    holdout_comp = pd.read_csv(MODEL_COMPARISON_PATH)

    print(f"Elapsed Time: {elapsed:.1f} seconds")
    print(f"Analytical Dataset: {ANALYTICAL_DATASET_PATH} ({len(df_analytical):,} rows)")
    print(f"Corporate Action Register: {CORPORATE_ACTION_REGISTER_PATH}")
    print(f"Fitted Best Model Pipeline: {BEST_MODEL_PATH}")
    print(f"Visualizations Saved: {FIGURES_DIR}")
    print(f"Reports Saved: {REPORTS_DIR}")

    print("\n--- Walk-Forward Validation Results (Mean across 3 folds) ---")
    for model_name in wf_summary.index:
        auc_mean = wf_summary.loc[model_name, ("roc_auc", "mean")]
        auc_std = wf_summary.loc[model_name, ("roc_auc", "std")]
        acc_mean = wf_summary.loc[model_name, ("accuracy", "mean")]
        print(f"  {model_name:25s} | ROC-AUC: {auc_mean:.4f} +/- {auc_std:.4f} | Accuracy: {acc_mean:.4f}")

    print("\n--- Final Chronological Holdout Evaluation (Untouched Test Window) ---")
    print(holdout_comp[["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

    print("\nSelected Architecture:", eval_results["best_model_name"])
    reg = eval_results["regression_results"]
    print(f"Secondary Regression Experiment MAE: {reg['mae']:.4f} "
          f"| RMSE: {reg['rmse']:.4f} "
          f"| OOS R² (vs train-mean): {reg['r2_vs_train_mean']:.4f}"
          f"\n  Directional Acc: {reg['directional_accuracy']:.4f}"
          f" | Naive Zero-Return MAE: {reg['naive_zero_mae']:.4f}"
          f" | Naive Train-Mean MAE: {reg['naive_mean_mae']:.4f}")

    print("\nTo generate predictions on latest data, run:")
    print("  python predict.py --symbol RELIANCE")
    print("  python predict.py --all")
    print("=" * 76)


if __name__ == "__main__":
    run_full_pipeline()
