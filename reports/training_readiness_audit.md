# Quantitative Training Readiness Audit

**Project:** Indian NSE Stock Prediction using Market Data and Fundamental Data  
**Repository:** `https://github.com/saagarrao2005-web/IML_project`  
**Audit Timestamp:** 27 October 2026  
**Auditor:** Senior Quantitative Researcher, ML Systems Engineer & Financial Data Validation Specialist  
**Branch:** `prepare/training-pipeline`  

---

## 1. Executive Summary & Readiness Verdict

### **Readiness Verdict: CONDITIONALLY READY**

The quantitative machine learning pipeline has been rigorously audited, repaired, restructured, and verified against frozen experimental specifications. The training infrastructure is strictly prepared for model execution with zero data leakage.

| Evaluation Track | Status | Detailed Finding |
| :--- | :--- | :--- |
| **Task A: 5-Session Classification (Market Features)** | **READY** | 16 equities, 19,840 rows, 4 verified corporate actions, 0 invalid dates, 0 duplicates, 0 OHLC violations. Leak-free chronological walk-forward CV and holdout protocol ready. |
| **Task B: 3-Month Regression (Market Features)** | **READY** | $H=63$ forward returns constructed, naive baselines (Zero, Train-Mean) and regressors (Ridge, RF, HistGBR) implemented, 63-session purge buffer enforced. |
| **Fundamentals-Only Track** | **BLOCKED** | **0 of 16 equities** have raw financial filings committed in the workspace. No fundamental values have been fabricated. |
| **Combined Feature Track (Market + Fundamentals)** | **BLOCKED** | Matched observation requirements cannot be met without raw financial filings. Deferred pending filing ingestion. |

> [!IMPORTANT]
> In strict accordance with the prompt mandate, **final production models have NOT been trained yet**. The repository contains verified baseline code, test fixtures, and verified pipeline architectures ready for the formal training phase.

---

## 2. Repository Inventory & Inspection

### 2.1 Workspace Structure
```text
Stock_Prediction/
├── Datasets/                 # 16 Raw NSE Bhavcopy CSV files (2021-10-25 to 2026-10-23)
├── cleaned_Dataset/          # 16 Cleaned daily equity CSV files
├── reports/                  # Comprehensive audit reports and deliverables
│   ├── training_readiness_audit.md
│   ├── data_validation_report.md
│   ├── training_pipeline_design.md
│   └── reproducibility_guide.md
├── research_protocol.md      # Frozen quantitative research specifications
├── requirements.txt          # Root Python package dependencies
├── app.py                    # Root Streamlit forwarding entrypoint
└── model_development/        # Core quantitative engineering pipeline
    ├── config.py             # Central configuration & corporate action register
    ├── experiment_config.py  # Frozen experimental constants & feature schemas
    ├── data_pipeline.py      # Ingestion, adjustment, feature & target generation
    ├── validate_data.py      # Standalone reproducible data & provenance auditor
    ├── training_pipeline.py  # Modular leakage-free multi-model training pipeline
    ├── train_evaluate.py     # Legacy evaluation and plotting script
    ├── predict.py            # Prediction CLI & programmatic inference interface
    ├── run_project.py        # Master pipeline execution script
    ├── tests/
    │   └── test_pipeline.py  # 23 automated pytest verification tests (100% pass)
    └── outputs/              # Datasets, reports, figures, models, predictions
```

### 2.2 Reused vs. Newly Implemented Functionality

| Pipeline Component | Previous State | Current Readiness State | Notes |
| :--- | :--- | :--- | :--- |
| `data_pipeline.py` | 5-day target only | Enhanced for Task A ($H=5$) and Task B ($H=63$) | 1008 tail NaNs for 63d, 80 for 5d |
| `validate_data.py` | Scattered exploratory scripts | Standalone reproducible validation script | Machine-readable JSON output |
| `training_pipeline.py` | Hardcoded fold ratios | Modular pipeline with Ridge, RF, HistGBR, Baselines | Full support for Rank IC, OOS R² |
| `experiment_config.py` | Missing | Implemented | Frozen experimental definitions |
| `research_protocol.md` | Absent | Formally codified & frozen | Strict task separation |
| Test Suite | 10 tests | **23 comprehensive automated tests** | All 23 passing |

---

## 3. Data Integrity & Provenance Audit Summary

1. **Volume & Row Counts:**
   - Total rows: 19,840 uniform daily observations.
   - 16 distinct equities, each containing exactly 1,240 trading sessions.
   - Zero missing dates, zero out-of-sequence calendar timestamps.
2. **Pricing & Trading Bounds:**
   - 0 duplicate `(Symbol, Date)` keys across the entire dataset.
   - 0 invalid or `NaN` trading dates.
   - 0 non-positive closing prices.
   - 0 OHLC logical violations ($H \ge L$, $H \ge O$, $H \ge C$, $L \le O$, $L \le C$).
   - 0 zero-volume or negative trading volume records.
3. **Corporate Actions Backward Adjustment:**
   - 4 confirmed mechanical price actions reconciled:
     - `MOTILALOFS` (Bonus 3:1 on 2024-06-10, multiplier $m=0.25$)
     - `RELIANCE` (Bonus 1:1 on 2024-10-28, multiplier $m=0.50$)
     - `SIEMENS` (Demerger on 2025-04-07, multiplier $m=0.49713$)
     - `HDFCBANK` (Bonus 1:1 on 2025-08-26, multiplier $m=0.50$)
   - Adjusted overnight returns across ex-dates are strictly within normal bounds ($|R| < 20\%$).
   - Rupee turnover ($P \times V$) remains strictly invariant across adjustments.

---

## 4. Fundamental Data Coverage Verification

### 4.1 Verification of External Claims
- **Claim:** *"Only 3 of 16 stocks have fundamental filings and HDFCBANK has an extraction problem."*
- **Empirical Repository Verification:**
  - An exhaustive scan of the repository tree reveals **0 raw financial filings** (XBRL, HTML, PDF) and **0 fundamental metric columns** in the committed CSV datasets.
  - The datasets committed in `cleaned_Dataset/` and `Datasets/` are exclusively daily NSE Bhavcopy trade records.
  - No raw financial statements for HDFC Bank or any other stock are present in the repository.
- **Protocol Adherence:**
  - Section 3 of the mandate strictly prohibits fabricating financial data or silently filling missing values with invented fundamentals.
  - The fundamental coverage gap is officially documented: **0% coverage across all 16 equities**.
  - As a result, the Fundamentals-Only and Combined feature tracks are classified as **BLOCKED pending data provision**.

---

## 5. Leakage Prevention & Preprocessing Verification

1. **Feature Construction:**
   - All 20 market features are computed strictly from data finalized at or before $15:30$ IST on session $t$.
   - Forward shifts are applied exclusively to target construction, never to predictor features.
   - Warmup-period missing values in rolling indicators (e.g. 50-day SMA, 20-day volatility) are preserved as `NaN` for the training-fold imputer rather than pre-filled with look-ahead constants.
2. **Preprocessing Isolation:**
   - Imputers (`SimpleImputer(strategy="median")`) and scalers (`StandardScaler()`) are encapsulated inside `Pipeline` estimators.
   - Preprocessors are fitted strictly on training fold observations. Transforming validation or holdout data uses exclusively training statistics.
3. **Purging & Embargo Integrity:**
   - Task A: 5-trading-session purge window between training and validation/test splits.
   - Task B: 63-trading-session purge window between training and validation/test splits.
   - Proof: Training target end date ($t_{\text{train}} + H$) strictly precedes evaluation start date ($t_{\text{eval}}$).

---

## 6. Readiness Verification Test Results

```bash
python -m pytest tests/ -v
```
**Results:**
- `test_symbol_inventory`: **PASSED**
- `test_date_parsing_and_chronology`: **PASSED**
- `test_symbol_date_uniqueness`: **PASSED**
- `test_eq_series_enforcement`: **PASSED**
- `test_ohlc_consistency`: **PASSED**
- `test_corporate_action_adjustments`: **PASSED**
- `test_target_alignment_and_no_symbol_bleed`: **PASSED**
- `test_no_future_leakage_in_features`: **PASSED**
- `test_chronological_purge_separation`: **PASSED**
- `test_prediction_interface_schema`: **PASSED**
- `test_no_inf_in_features`: **PASSED**
- `test_feature_columns_exist`: **PASSED**
- `test_warmup_nans_only_at_start`: **PASSED**
- `test_walk_forward_uses_config_specs`: **PASSED**
- `test_holdout_predictions_exist_and_consistent`: **PASSED**
- `test_evaluation_report_json_exists`: **PASSED**
- `test_class_balance_per_symbol`: **PASSED**
- `test_model_serialization_fidelity`: **PASSED**
- `test_task_b_target_63d_alignment`: **PASSED**
- `test_regression_pipeline_models`: **PASSED**
- `test_preprocessing_isolation_no_leakage`: **PASSED**
- `test_fundamental_coverage_transparency`: **PASSED**
- `test_purge_window_task_b_sufficiency`: **PASSED**

**Summary: 23 passed, 0 failed (100% pass rate).**

---

## 7. Blocker Summary & Final Recommendation

1. **Market-Data Training:** The pipeline is **100% READY** for training all candidate models across Task A (Directional Classification) and Task B (Continuous Return Regression) using the 20 market features.
2. **Fundamental-Data Training:** **BLOCKED** until verified, point-in-time financial statements (with official filing timestamps) are supplied for the 16 equities.
3. **Action Plan:** Proceed to training the market-only baseline and candidate models under the frozen research protocol.
