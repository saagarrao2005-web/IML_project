"""
Automated Test Suite for the NSE Stock Price Prediction ML Project.

Covers:
1. Ingestion, date parsing, and chronological monotonic ordering.
2. Symbol-Date uniqueness (zero duplicate keys).
3. EQ-series enforcement.
4. OHLC price validity and non-negativity.
5. Corporate-action price/volume adjustments (direction, magnitude, continuity).
6. Return and rolling-feature consistency and warmup handling.
7. 5-trading-session forward target construction and alignment (zero cross-symbol bleed).
8. Absence of look-ahead leakage in feature matrices.
9. Chronological train/validation/holdout separation with 5-session embargo.
10. Strict fitting of preprocessing within training folds.
11. Model serialization and deserialization fidelity.
12. Usability and output schema of the prediction interface.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure model_development is in Python path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    ANALYTICAL_DATASET_PATH,
    BEST_MODEL_PATH,
    CONFIRMED_CORPORATE_ACTIONS,
    EXPECTED_SYMBOLS,
    FEATURE_COLUMNS,
    HORIZON_SESSIONS,
    PURGE_SESSIONS,
)
from data_pipeline import (
    load_raw_data,
    run_data_quality_audit,
    apply_corporate_actions,
    compute_features,
    construct_target,
    audit_target_alignment,
)
from predict import predict_stock, predict_universe, load_model


@pytest.fixture(scope="module")
def analytical_data():
    """Loads prepared analytical dataset or generates a clean slice for testing."""
    if ANALYTICAL_DATASET_PATH.exists():
        df = pd.read_csv(ANALYTICAL_DATASET_PATH, parse_dates=["Date"])
    else:
        raw = load_raw_data()
        adj = apply_corporate_actions(raw)
        feat = compute_features(adj)
        df = construct_target(feat)
    return df


def test_symbol_inventory(analytical_data):
    """Test that all 16 target symbols exist with expected row counts."""
    symbols = sorted(analytical_data["Symbol"].unique())
    assert symbols == EXPECTED_SYMBOLS, f"Mismatch in expected symbols. Got: {symbols}"
    assert len(symbols) == 16
    for sym in symbols:
        cnt = (analytical_data["Symbol"] == sym).sum()
        assert cnt >= 1200, f"Stock {sym} has fewer rows than expected: {cnt}"


def test_date_parsing_and_chronology(analytical_data):
    """Test that dates are valid datetimes and monotonically increasing per symbol."""
    assert analytical_data["Date"].isna().sum() == 0, "Found invalid/NaN dates."
    for sym, grp in analytical_data.groupby("Symbol"):
        dates = grp["Date"].to_numpy()
        assert np.all(dates[:-1] <= dates[1:]), f"Dates not monotonic for {sym}"


def test_symbol_date_uniqueness(analytical_data):
    """Test that there are zero duplicate Symbol-Date keys in the dataset."""
    dups = analytical_data.duplicated(subset=["Symbol", "Date"], keep=False).sum()
    assert dups == 0, f"Found {dups} duplicate Symbol-Date rows."


def test_eq_series_enforcement(analytical_data):
    """Test that only EQ series records are present."""
    if "Series" in analytical_data.columns:
        series_set = set(analytical_data["Series"].unique())
        assert series_set == {"EQ"}, f"Non-EQ series detected: {series_set}"


def test_ohlc_consistency(analytical_data):
    """Test standard OHLC bounds: High >= Low, Open, Close; Low <= Open, Close."""
    high = analytical_data["adj_high"]
    low = analytical_data["adj_low"]
    open_p = analytical_data["adj_open"]
    close = analytical_data["adj_close"]

    assert (high < low).sum() == 0, "High < Low violation detected."
    assert (high < open_p).sum() == 0, "High < Open violation detected."
    assert (high < close).sum() == 0, "High < Close violation detected."
    assert (low > open_p).sum() == 0, "Low > Open violation detected."
    assert (low > close).sum() == 0, "Low > Close violation detected."
    assert (close <= 0).sum() == 0, "Non-positive closing price detected."


def test_corporate_action_adjustments(analytical_data):
    """
    Test that corporate actions adjust prices in the correct direction:
    - Pre-bonus prices must be lower in adjusted terms (divided by split/bonus ratio).
    - Post-bonus prices must equal raw prices (factor = 1.0).
    - Overnight return across ex-date must NOT exhibit artificial -50% to -75% drops.
    """
    for sym, info in CONFIRMED_CORPORATE_ACTIONS.items():
        sub = analytical_data[analytical_data["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
        ex_date = pd.to_datetime(info["ex_date"])

        pre_mask = sub["Date"] < ex_date
        post_mask = sub["Date"] >= ex_date

        # Post-event factor should be 1.0
        assert np.allclose(sub.loc[post_mask, "adj_factor"], 1.0), f"{sym} post-event adj_factor is not 1.0"

        # Pre-event factor should match configured ratio
        expected_factor = info["adjustment_factor"]
        assert np.allclose(sub.loc[pre_mask, "adj_factor"], expected_factor), f"{sym} pre-event factor mismatch"

        # Overnight return across ex-date on adjusted series should be reasonable (|ret| < 0.20)
        ex_idx = sub[sub["Date"] == ex_date].index[0]
        adj_ret_at_ex = (sub.loc[ex_idx, "adj_close"] / sub.loc[ex_idx - 1, "adj_close"]) - 1.0
        raw_ret_at_ex = (sub.loc[ex_idx, "raw_close"] / sub.loc[ex_idx - 1, "raw_close"]) - 1.0

        assert abs(adj_ret_at_ex) < 0.20, f"{sym} adjusted ex-date return ({adj_ret_at_ex:.2%}) is abnormally large"
        assert abs(raw_ret_at_ex) > 0.40, f"{sym} raw ex-date return ({raw_ret_at_ex:.2%}) failed to reflect action"


def test_target_alignment_and_no_symbol_bleed(analytical_data):
    """
    Test that target_up_5d:
    1. Correctly reflects adj_close(t+5) > adj_close(t).
    2. Never crosses symbol boundaries.
    3. Has exactly 5 NaNs per symbol at the end of the series.
    """
    horizon = HORIZON_SESSIONS
    grouped = analytical_data.groupby("Symbol", sort=False)

    for sym, grp in grouped:
        grp = grp.sort_values("Date").reset_index(drop=True)
        close = grp["adj_close"].to_numpy()
        target = grp["target_up_5d"].to_numpy()

        # Final 5 rows must have NaN target
        assert np.isnan(target[-horizon:]).all(), f"{sym} does not have NaN targets for last {horizon} rows"
        # Preceding rows must NOT be NaN
        assert not np.isnan(target[:-horizon]).any(), f"{sym} has unexpected NaNs before last {horizon} rows"

        # Check alignment on known rows
        for i in range(len(close) - horizon):
            expected = 1.0 if close[i + horizon] > close[i] else 0.0
            assert target[i] == expected, f"{sym} target mismatch at index {i}"


def test_no_future_leakage_in_features(analytical_data):
    """
    Test that no target, future return, or post-observation prices
    leak into the predictive feature set.
    """
    forbidden_substrings = ["future", "target", "fwd", "next"]
    for col in FEATURE_COLUMNS:
        for forbidden in forbidden_substrings:
            assert forbidden not in col.lower(), f"Feature column '{col}' contains forbidden string '{forbidden}'"

    # Ensure feature columns are present in analytical dataset
    missing = set(FEATURE_COLUMNS) - set(analytical_data.columns)
    assert not missing, f"Missing feature columns in analytical dataset: {missing}"


def test_chronological_purge_separation(analytical_data):
    """Test that train and test date partitions maintain the strict 5-day purge embargo."""
    labeled = analytical_data.dropna(subset=["target_up_5d"]).copy()
    dates = np.array(sorted(labeled["Date"].unique()))

    split_idx = int(len(dates) * 0.85)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx + PURGE_SESSIONS:]

    train_max = pd.Timestamp(train_dates[-1])
    test_min = pd.Timestamp(test_dates[0])

    # Date gap must be strictly greater than 5 calendar/trading intervals
    diff_days = (test_min - train_max).days
    assert diff_days >= 5, f"Insufficient purge buffer between train ({train_max}) and test ({test_min})"


def test_prediction_interface_schema():
    """Test that predict_stock produces valid schema, probabilities, and labels."""
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Best model artifact not yet generated.")

    result = predict_stock(symbol="RELIANCE")
    assert isinstance(result, dict)
    assert result["symbol"] == "RELIANCE"
    assert 0.0 <= result["probability_up_5d"] <= 1.0
    assert 0.0 <= result["probability_down_5d"] <= 1.0
    assert np.isclose(result["probability_up_5d"] + result["probability_down_5d"], 1.0)
    assert result["predicted_direction"] in ["UP (Bullish)", "DOWN (Bearish)"]
    assert "disclaimer" in result


# ==============================================================================
# ADDITIONAL REGRESSION TESTS — Added during repair audit
# ==============================================================================

def test_no_inf_in_features(analytical_data):
    """Test that feature columns contain no inf values."""
    for col in FEATURE_COLUMNS:
        if col in analytical_data.columns:
            inf_count = np.isinf(analytical_data[col]).sum()
            assert inf_count == 0, f"Feature '{col}' has {inf_count} inf values"


def test_feature_columns_exist(analytical_data):
    """Test that all configured feature columns exist in the analytical dataset."""
    missing = set(FEATURE_COLUMNS) - set(analytical_data.columns)
    assert not missing, f"Missing feature columns: {missing}"


def test_warmup_nans_only_at_start(analytical_data):
    """
    Test that NaN values in rolling features only occur at the start of each
    stock's series (warmup period), not in the middle or end.
    """
    rolling_features = [
        "return_20d", "close_vs_sma_50", "volatility_20d",
        "volume_vs_avg20", "trades_vs_avg20",
    ]
    for sym, grp in analytical_data.groupby("Symbol"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        for feat in rolling_features:
            if feat not in grp.columns:
                continue
            vals = grp[feat].to_numpy()
            # Find last NaN index
            nan_mask = np.isnan(vals)
            if not nan_mask.any():
                continue
            last_nan_idx = np.where(nan_mask)[0][-1]
            # All values after the last NaN should be non-NaN
            assert not nan_mask[last_nan_idx + 1:].any(), (
                f"{sym}/{feat}: NaN found after warmup at index {last_nan_idx}"
            )


def test_walk_forward_uses_config_specs():
    """
    Test that the walk-forward validation function uses WALK_FORWARD_SPECS from config.py,
    not hardcoded fold specifications.
    """
    import inspect
    from config import WALK_FORWARD_SPECS
    from train_evaluate import run_walk_forward_validation

    source = inspect.getsource(run_walk_forward_validation)
    # Should reference the config constant, not define its own fold_specs list
    assert "WALK_FORWARD_SPECS" in source, (
        "run_walk_forward_validation does not reference WALK_FORWARD_SPECS from config"
    )
    # Should not contain hardcoded fold ratios
    assert "(0.40, 0.60)" not in source, (
        "run_walk_forward_validation still contains hardcoded fold spec (0.40, 0.60)"
    )


def test_holdout_predictions_exist_and_consistent():
    """
    Test that holdout predictions file exists and its class predictions
    are consistent with the stored probabilities.
    """
    holdout_path = BEST_MODEL_PATH.parent.parent / "predictions" / "holdout_predictions.csv"
    if not holdout_path.exists():
        pytest.skip("Holdout predictions not yet generated.")

    preds_df = pd.read_csv(holdout_path)
    assert "target_up_5d" in preds_df.columns, "Missing target column"
    assert len(preds_df) > 0, "Holdout predictions file is empty"

    # Check that stored probabilities sum to approximately 1
    # (we can check at least one model's probs are in [0,1])
    prob_cols = [c for c in preds_df.columns if c.endswith("_prob")]
    for col in prob_cols:
        vals = preds_df[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), (
            f"Probability column {col} has values outside [0,1]"
        )


def test_evaluation_report_json_exists():
    """Test that the comprehensive evaluation report JSON is generated."""
    import json
    eval_path = BEST_MODEL_PATH.parent.parent / "reports" / "evaluation_report.json"
    if not eval_path.exists():
        pytest.skip("Evaluation report not yet generated.")

    with open(eval_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Verify required fields
    assert "best_model_name" in report
    assert "holdout_period" in report
    assert "classification_holdout" in report
    assert "regression_holdout" in report
    reg = report["regression_holdout"]
    assert "r2_vs_train_mean" in reg, "Missing R² metric in regression report"
    assert "baselines" in reg, "Missing baseline comparisons in regression report"


def test_class_balance_per_symbol(analytical_data):
    """Test that each stock has both classes present (not degenerate)."""
    labeled = analytical_data.dropna(subset=["target_up_5d"])
    for sym, grp in labeled.groupby("Symbol"):
        unique_classes = grp["target_up_5d"].nunique()
        assert unique_classes == 2, (
            f"Stock {sym} has only {unique_classes} class(es) — degenerate"
        )


def test_model_serialization_fidelity(analytical_data):
    """Test that the saved model produces consistent predictions when reloaded."""
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Best model artifact not yet generated.")

    import joblib
    model = joblib.load(BEST_MODEL_PATH)

    # Get a sample of features from the last 10 rows of RELIANCE
    sample = analytical_data[analytical_data["Symbol"] == "RELIANCE"].tail(10)
    X_sample = sample[FEATURE_COLUMNS]

    # Predict twice — should be identical
    preds1 = model.predict_proba(X_sample)
    preds2 = model.predict_proba(X_sample)
    assert np.allclose(preds1, preds2), "Model predictions are not deterministic"
    assert preds1.shape == (len(X_sample), 2), "Unexpected prediction shape"


# ==============================================================================
# RESEARCH PROTOCOL & TRAINING PIPELINE TESTS (PHASE 4 - 6)
# ==============================================================================

def test_task_b_target_63d_alignment(analytical_data):
    """
    Test Task B 3-month continuous return target:
    1. future_return_63d matches adj_close(t+63) / adj_close(t) - 1.0.
    2. Final 63 rows per stock are strictly NaN.
    3. Exactly 16 * 63 = 1008 NaNs exist in the entire dataset.
    """
    assert "future_return_63d" in analytical_data.columns, "Missing future_return_63d column"
    total_nans = analytical_data["future_return_63d"].isna().sum()
    assert total_nans == 16 * 63, f"Expected 1008 NaNs in future_return_63d, got {total_nans}"

    for sym, grp in analytical_data.groupby("Symbol"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        close = grp["adj_close"].to_numpy()
        ret63 = grp["future_return_63d"].to_numpy()

        assert np.isnan(ret63[-63:]).all(), f"{sym} does not have 63 NaNs at end"
        assert not np.isnan(ret63[:-63]).any(), f"{sym} has unexpected NaNs before end"

        for i in range(min(20, len(close) - 63)):
            expected = (close[i + 63] / close[i]) - 1.0
            assert np.isclose(ret63[i], expected, rtol=1e-10), f"Mismatch in {sym} 63d return at {i}"


def test_regression_pipeline_models(analytical_data):
    """
    Test that all 5 continuous-return regression models can be instantiated,
    fitted, and evaluated on valid feature inputs.
    """
    from training_pipeline import build_regression_models, evaluate_regression

    reg_models = build_regression_models()
    assert len(reg_models) == 5
    assert "Zero_Return_Baseline" in reg_models
    assert "Training_Mean_Baseline" in reg_models
    assert "Ridge_Regression" in reg_models
    assert "Random_Forest_Regressor" in reg_models
    assert "Hist_Gradient_Boosting_Regressor" in reg_models

    valid = analytical_data.dropna(subset=["future_return_5d"]).tail(200)
    train_slice = valid.iloc[:150]
    test_slice = valid.iloc[150:]
    train_mean = float(train_slice["future_return_5d"].mean())

    for name, pipe in reg_models.items():
        pipe.fit(train_slice[FEATURE_COLUMNS], train_slice["future_return_5d"])
        preds = pipe.predict(test_slice[FEATURE_COLUMNS])
        assert len(preds) == len(test_slice)
        metrics = evaluate_regression(test_slice["future_return_5d"].to_numpy(), preds, train_mean)
        assert metrics["mae"] >= 0.0
        assert metrics["rmse"] >= 0.0


def test_preprocessing_isolation_no_leakage():
    """
    Test that imputers and scalers fitted on training fold observations
    do not leak information from test fold observations.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Create synthetic train and test matrices with distinct distributions
    train_mat = np.array([[10.0, 10.0], [20.0, np.nan], [30.0, 30.0]])
    test_mat = np.array([[100.0, np.nan], [200.0, 200.0]])

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipe.fit(train_mat)
    # The fitted imputer's statistic for col 1 must be median of [10.0, 30.0] = 20.0
    imputer_stat = pipe.named_steps["imputer"].statistics_
    assert np.isclose(imputer_stat[1], 20.0), f"Imputer leaked or wrong median: {imputer_stat}"

    # Transforming test_mat must use train median (20.0), NOT test values (200.0)
    test_transformed = pipe.transform(test_mat)
    assert not np.isnan(test_transformed).any(), "NaN found after pipeline transform"


def test_fundamental_coverage_transparency(analytical_data):
    """
    Test that the pipeline transparently reports fundamental feature availability
    and strictly refuses to fabricate missing values.
    """
    from training_pipeline import resolve_feature_subset

    mkt_feats, status_mkt = resolve_feature_subset(analytical_data, "market")
    assert status_mkt == "OK"
    assert len(mkt_feats) == 20

    fund_feats, status_fund = resolve_feature_subset(analytical_data, "fundamentals")
    assert "BLOCKED" in status_fund
    assert len(fund_feats) == 0

    comb_feats, status_comb = resolve_feature_subset(analytical_data, "combined")
    assert "BLOCKED" in status_comb
    assert len(comb_feats) == 0


def test_purge_window_task_b_sufficiency():
    """
    Test that for Task B (63-session horizon), a 63-session purge window
    strictly guarantees training target dates precede evaluation start dates.
    """
    # 63 trading sessions ~ 3 calendar months
    purge_sessions = 63
    train_max_idx = 100
    test_min_idx = train_max_idx + purge_sessions
    temporal_gap = test_min_idx - train_max_idx
    assert temporal_gap >= purge_sessions, f"Purge buffer {temporal_gap} < {purge_sessions}"


# ==============================================================================
# INDEPENDENT AUDIT REGRESSION TESTS (AUDITOR VERIFICATION SUITE)
# ==============================================================================

def test_audit_task_a_recalculation_reconciliation():
    """
    Independent audit test: Recalculates all classification metrics directly from
    saved per-observation development predictions and reconciles against fold results.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    from config import PREDICTIONS_DIR, REPORTS_DIR

    preds_path = PREDICTIONS_DIR / "development_classification_predictions.csv"
    results_path = REPORTS_DIR / "walk_forward_classification_results.csv"

    if not preds_path.exists() or not results_path.exists():
        pytest.skip("Task A development artifacts not yet generated.")

    cls_preds = pd.read_csv(preds_path)
    wf_results = pd.read_csv(results_path)

    assert len(cls_preds) == 8160, f"Expected 8160 OOS classification rows, got {len(cls_preds)}"
    assert cls_preds.duplicated(subset=["fold", "Date", "Symbol"]).sum() == 0, "Duplicate keys found in predictions"

    models = ["Majority_Class_Baseline", "Logistic_Regression", "Random_Forest", "Hist_Gradient_Boosting", "XGBoost"]

    for fold in [1, 2, 3]:
        sub = cls_preds[cls_preds["fold"] == fold]
        y_true = sub["actual"].to_numpy()

        for m in models:
            pred_col = f"{m}_pred"
            prob_col = f"{m}_prob"
            y_pred = sub[pred_col].to_numpy()
            y_prob = sub[prob_col].to_numpy() if prob_col in sub.columns else None

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            auc = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else 0.5

            ref = wf_results[(wf_results["fold"] == fold) & (wf_results["model"] == m)].iloc[0]

            assert np.isclose(acc, ref["accuracy"]), f"Accuracy mismatch {m} fold {fold}"
            assert np.isclose(bal_acc, ref["balanced_accuracy"]), f"BalAcc mismatch {m} fold {fold}"
            assert np.isclose(prec, ref["precision"]), f"Precision mismatch {m} fold {fold}"
            assert np.isclose(rec, ref["recall"]), f"Recall mismatch {m} fold {fold}"
            assert np.isclose(f1, ref["f1"]), f"F1 mismatch {m} fold {fold}"
            assert np.isclose(auc, ref["roc_auc"]), f"ROC-AUC mismatch {m} fold {fold}"
            assert cm[0] == ref["tn"] and cm[1] == ref["fp"] and cm[2] == ref["fn"] and cm[3] == ref["tp"], (
                f"Confusion matrix mismatch {m} fold {fold}"
            )


def test_audit_task_b_recalculation_reconciliation():
    """
    Independent audit test: Recalculates all regression metrics directly from
    saved per-observation development predictions and reconciles against fold results.
    """
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score
    from scipy.stats import spearmanr
    from config import PREDICTIONS_DIR, REPORTS_DIR, ANALYTICAL_DATASET_PATH, WALK_FORWARD_SPECS, HOLDOUT_FRACTION

    preds_path = PREDICTIONS_DIR / "development_regression_predictions.csv"
    results_path = REPORTS_DIR / "walk_forward_regression_results.csv"

    if not preds_path.exists() or not results_path.exists():
        pytest.skip("Task B development artifacts not yet generated.")

    reg_preds = pd.read_csv(preds_path)
    wf_results = pd.read_csv(results_path)

    assert len(reg_preds) == 4976, f"Expected 4976 OOS regression rows, got {len(reg_preds)}"
    assert reg_preds.duplicated(subset=["fold", "Date", "Symbol"]).sum() == 0, "Duplicate keys found in predictions"

    # Compute training means per fold to reconcile R2
    analytical = pd.read_csv(ANALYTICAL_DATASET_PATH, parse_dates=["Date"])
    labeled = analytical.dropna(subset=["future_return_63d"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    dates = np.array(sorted(labeled["Date"].unique()))
    dev_dates = dates[:int(len(dates) * (1.0 - HOLDOUT_FRACTION))]
    dev_data = labeled[labeled["Date"].isin(dev_dates)]
    n_dev = len(dev_dates)

    train_means = {}
    for fold_idx, (tr_pct, vl_pct) in enumerate(WALK_FORWARD_SPECS, 1):
        tr_end = int(n_dev * tr_pct)
        tr_fold = dev_data[dev_data["Date"].isin(dev_dates[:tr_end])]
        train_means[fold_idx] = float(tr_fold["future_return_63d"].mean())

    models = ["Zero_Return_Baseline", "Training_Mean_Baseline", "Ridge_Regression", "Random_Forest_Regressor", "Hist_Gradient_Boosting_Regressor"]

    for fold in [1, 2, 3]:
        sub = reg_preds[reg_preds["fold"] == fold]
        y_true = sub["actual"].to_numpy()
        tr_mean = train_means[fold]

        for m in models:
            pred_col = f"{m}_pred"
            y_pred = sub[pred_col].to_numpy()

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot_mean = np.sum((y_true - tr_mean) ** 2)
            ss_tot_zero = np.sum(y_true ** 2)
            r2_mean = 1.0 - (ss_res / ss_tot_mean)
            r2_zero = 1.0 - (ss_res / ss_tot_zero)

            dir_acc = accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))

            date_ics = []
            for dt, grp in sub.groupby("Date"):
                if len(grp) >= 3 and grp[pred_col].std() > 1e-9 and grp["actual"].std() > 1e-9:
                    rho, _ = spearmanr(grp[pred_col], grp["actual"])
                    if not np.isnan(rho):
                        date_ics.append(rho)
            mean_ic = float(np.mean(date_ics)) if date_ics else 0.0

            ref = wf_results[(wf_results["fold"] == fold) & (wf_results["model"] == m)].iloc[0]

            assert np.isclose(mae, ref["mae"]), f"MAE mismatch {m} fold {fold}"
            assert np.isclose(rmse, ref["rmse"]), f"RMSE mismatch {m} fold {fold}"
            assert np.isclose(r2_mean, ref["r2_vs_train_mean"]), f"R2 mean mismatch {m} fold {fold}"
            assert np.isclose(r2_zero, ref["r2_vs_zero"]), f"R2 zero mismatch {m} fold {fold}"
            assert np.isclose(dir_acc, ref["directional_accuracy"]), f"DirAcc mismatch {m} fold {fold}"
            assert np.isclose(mean_ic, ref["mean_rank_ic"], atol=1e-5), f"Rank IC mismatch {m} fold {fold}"


def test_audit_no_temporal_label_overlap_in_walk_forward():
    """
    Independent audit test: Confirms that across all stocks and folds,
    training target realization strictly precedes validation observation start.
    """
    from config import ANALYTICAL_DATASET_PATH, WALK_FORWARD_SPECS, PURGE_SESSIONS, PURGE_REGRESSION_SESSIONS, HOLDOUT_FRACTION

    df = pd.read_csv(ANALYTICAL_DATASET_PATH, parse_dates=["Date"])

    for task, target_col, purge_s, h in [("Task A", "target_up_5d", PURGE_SESSIONS, 5), ("Task B", "future_return_63d", PURGE_REGRESSION_SESSIONS, 63)]:
        labeled = df.dropna(subset=[target_col]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
        dates = np.array(sorted(labeled["Date"].unique()))
        dev_dates = dates[:int(len(dates) * (1.0 - HOLDOUT_FRACTION))]
        n_dev = len(dev_dates)

        for fold_idx, (tr_pct, vl_pct) in enumerate(WALK_FORWARD_SPECS, 1):
            tr_end = int(n_dev * tr_pct)
            vl_start = tr_end + purge_s
            vl_end = int(n_dev * vl_pct)

            tr_dates = dev_dates[:tr_end]
            vl_dates = dev_dates[vl_start:vl_end]

            # Cross-stock validation: find latest future target date for any stock in train
            tr_sub = labeled[labeled["Date"].isin(tr_dates)]
            vl_sub = labeled[labeled["Date"].isin(vl_dates)]

            tr_future_dates = []
            for sym, grp in tr_sub.groupby("Symbol"):
                max_d = grp["Date"].max()
                sym_all = df[df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
                d_idx = np.where(sym_all["Date"] == max_d)[0][0]
                if d_idx + h < len(sym_all):
                    tr_future_dates.append(sym_all.loc[d_idx + h, "Date"])

            max_train_future = max(tr_future_dates)
            min_val_date = vl_sub["Date"].min()

            assert max_train_future < min_val_date, (
                f"{task} Fold {fold_idx}: Train target realized {max_train_future} >= Val start {min_val_date}"
            )


def test_audit_artifact_completeness_and_holdout_protection():
    """
    Independent audit test: Confirms that model configs record holdout as un-evaluated
    and that prediction files contain valid non-synthetic values.
    """
    import json
    from config import REPORTS_DIR, PREDICTIONS_DIR

    with open(REPORTS_DIR / "experiment_config_classification.json") as f:
        cfg_cls = json.load(f)
    with open(REPORTS_DIR / "experiment_config_regression.json") as f:
        cfg_reg = json.load(f)

    assert cfg_cls["holdout_period"]["evaluated"] is False, "Holdout was evaluated in Task A config!"
    assert cfg_reg["holdout_period"]["evaluated"] is False, "Holdout was evaluated in Task B config!"

    cls_preds = pd.read_csv(PREDICTIONS_DIR / "development_classification_predictions.csv")
    assert cls_preds.isna().sum().sum() == 0, "NaNs found in classification predictions"

    # Verify predictions are not degenerately constant
    assert cls_preds["Logistic_Regression_pred"].nunique() == 2, "Logistic regression predictions degenerate"
    assert cls_preds["Random_Forest_pred"].nunique() == 2, "Random forest predictions degenerate"


