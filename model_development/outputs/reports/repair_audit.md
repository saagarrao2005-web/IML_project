# Repair Audit Report

**Repository:** `d:\B.Tech\Sem3\IML\Project\Stock_Prediction`
**Audit Date:** 27 October 2026
**Branch:** `repair/audit-and-fix`
**Baseline Commit:** `cd415de` (main)

---

## Executive Summary

Six confirmed issues were identified and fixed. All 18 automated tests pass after repair. The pipeline is fully reproducible. No data was deleted or fabricated. No fundamental data exists in this repository, so fundamental-related evaluation tasks (Section 3, 5B of the original task) are documented as not applicable.

---

## Confirmed Issues and Fixes

### Issue 1: Walk-Forward Fold Specs Mismatch (SEVERITY: HIGH)

**Problem:** `config.py` defines `WALK_FORWARD_SPECS = [(0.50, 0.65), (0.65, 0.80), (0.80, 1.00)]` but `train_evaluate.py` hardcodes different fold specs `[(0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]`. The config value was dead code — never imported or used.

**Impact:** Walk-forward folds used different train/val proportions than documented. Fold 1 used 40%/60% instead of 50%/65%. This affected model selection and all walk-forward metrics in the original reports.

**Fix:** Modified `train_evaluate.py` to import and use `WALK_FORWARD_SPECS` from `config.py`.

**Files Changed:** `train_evaluate.py` (import + fold_specs assignment)

**Verification:** New test `test_walk_forward_uses_config_specs` confirms the config constant is used and hardcoded values are absent.

---

### Issue 2: Missing Regression Baselines and R² (SEVERITY: MEDIUM)

**Problem:** The regression experiment reported MAE and RMSE but did not compute:
- Out-of-sample R² against any benchmark
- Comparison with naive zero-return predictor
- Comparison with training-mean predictor

Without baselines, the regression MAE of 0.0347 appears reasonable but is actually *worse* than simply predicting zero (MAE 0.0339).

**Fix:** Added R² computation (vs train-mean and vs zero benchmarks), naive zero-return baseline, and naive training-mean baseline to `train_evaluate.py`. Results are now saved in `evaluation_report.json`.

**Files Changed:** `train_evaluate.py` (regression experiment section + return dict + JSON save)

---

### Issue 3: No Evaluation Report JSON Saved (SEVERITY: MEDIUM)

**Problem:** The holdout evaluation computed all metrics but did not persist them as a comprehensive machine-readable artifact. The dashboard and downstream scripts had to re-derive or hardcode values.

**Fix:** Added `evaluation_report.json` output containing: best model name, holdout/training periods, all classification holdout metrics, all regression metrics with baselines.

**Files Changed:** `train_evaluate.py`, verified by `test_evaluation_report_json_exists`

---

### Issue 4: Dashboard Hardcoded Model Name and Display Bugs (SEVERITY: MEDIUM)

**Problem:**
- Sidebar displayed "Model: Logistic Regression (L2)" as a hardcoded string
- Dashboard KPI card displayed "Deployed Model: Logistic Reg." as a hardcoded string
- Model Performance page hardcoded `comp_df[comp_df["model"] == "Logistic_Regression"]`
- Volume display showed `volume_vs_avg20` as "0.50x" when value is (V/avg − 1), meaning 0.50 represents 50% above average, not half

**Fix:**
- Added `get_deployed_model_name()` function that reads from `evaluation_report.json` or falls back to walk-forward selection
- All hardcoded model name references replaced with dynamic lookup
- Volume display corrected: now shows `(vol_ratio + 1.0)` as multiplier with percentage clarification

**Files Changed:** `app.py` (4 locations)

---

### Issue 5: RSI Warmup Fill with Neutral Value (SEVERITY: LOW)

**Problem:** RSI computation filled warmup-period NaN with 50.0 (neutral) before the pipeline imputer could handle it. This was inconsistent with the documented approach ("Median imputed in training fold") and masked NaN from the imputer's view. While not technically a future-data leak, it injected an arbitrary value that the imputer would have handled differently.

**Fix:** Removed `fillna(50.0)` — RSI warmup NaN values are now left for the pipeline's `SimpleImputer(strategy="median")` to handle within training folds, consistent with all other features.

**Files Changed:** `data_pipeline.py` (RSI function)

**Note:** This was already fixed in the `data_pipeline.py` on disk (likely from a prior partial repair). Confirmed the fix is in place.

---

### Issue 6: run_project.py Not Showing New Metrics (SEVERITY: LOW)

**Problem:** The executive summary output from `run_project.py` only showed MAE and directional accuracy for the regression experiment, not the new R² and baseline comparisons.

**Fix:** Updated the summary print statements to include R², naive zero MAE, and naive train-mean MAE.

**Files Changed:** `run_project.py`

---

## Issues Investigated but Not Confirmed as Bugs

| Investigated Item | Finding |
|:---|:---|
| pct_change fill_method deprecation | `fill_method=None` is the correct modern usage; no warning emitted |
| XGBoost missing class_weight | XGBoost uses `scale_pos_weight`, not `class_weight`. The omission is intentional — not all models need identical balancing |
| Root app.py double execution | `main()` is a function call, not module-level code. Streamlit re-runs on import are handled correctly |
| Fundamental data coverage | No fundamental data exists in this repository. The task references `research_protocol.md` which does not exist. This is a scope limitation, not a bug |
| Survivorship bias | Acknowledged limitation in existing reports. The 16 equities are surviving large-caps as of 2026. Not fixable without additional data |

---

## New Tests Added

| # | Test Name | Purpose |
|:---|:---|:---|
| 11 | `test_no_inf_in_features` | Verifies no inf values in feature columns |
| 12 | `test_feature_columns_exist` | Verifies all config features exist in analytical dataset |
| 13 | `test_warmup_nans_only_at_start` | Verifies NaN only occurs during warmup, not mid-series |
| 14 | `test_walk_forward_uses_config_specs` | Regression test for the fold specs bug |
| 15 | `test_holdout_predictions_exist_and_consistent` | Verifies predictions file has valid probabilities |
| 16 | `test_evaluation_report_json_exists` | Verifies the new evaluation JSON is generated with required fields |
| 17 | `test_class_balance_per_symbol` | Verifies each stock has both classes (non-degenerate) |
| 18 | `test_model_serialization_fidelity` | Verifies saved model produces deterministic predictions |

**All 18 tests pass.**

---

## Files Changed Summary

| File | Changes |
|:---|:---|
| `model_development/train_evaluate.py` | Fixed fold specs import, added regression baselines/R², added evaluation JSON save |
| `model_development/app.py` | Dynamic model name, volume display fix, REPORTS_DIR import |
| `model_development/data_pipeline.py` | RSI warmup fix (confirmed already in place) |
| `model_development/run_project.py` | Updated summary output for new regression metrics |
| `model_development/tests/test_pipeline.py` | Added 8 new regression tests |
| `model_development/verify_metrics.py` | Independent metric verification script (new) |
| `model_development/outputs/*` | Regenerated artifacts with corrected pipeline |

---

## Reproduction Commands

```bash
cd model_development

# Run full pipeline (data + training + evaluation + tests)
python run_project.py

# Run tests only
python -m pytest tests/ -v

# Independent metric verification
python verify_metrics.py

# Generate predictions
python predict.py --all

# Launch dashboard
streamlit run app.py
```
