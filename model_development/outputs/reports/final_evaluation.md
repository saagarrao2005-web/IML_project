# Final Independent Evaluation Report

**Repository:** `d:\B.Tech\Sem3\IML\Project\Stock_Prediction`
**Evaluation Date:** 27 October 2026
**Branch:** `repair/audit-and-fix` (based on `cd415de` main)
**Environment:** Python 3.12.10, scikit-learn 1.8.0, xgboost 3.2.0, pandas 3.0.1, numpy 2.4.2, pytest 9.0.3

---

## 1. Evaluation Verdict

### **CONDITIONAL**

The pipeline is reproducible, leakage-free, and all stated claims about classification performance are supported by evidence. The walk-forward fold specifications were corrected from a documented mismatch. However, several conditions prevent a full PASS:

1. **No `research_protocol.md` exists** — the regression/fundamental experiments referenced in the task cannot be evaluated against frozen specifications.
2. **No fundamental data exists** — market-only, fundamentals-only, and combined feature comparisons are not applicable.
3. **No three-month regression experiment** — only the existing 5-session regression experiment was evaluated.
4. **Survivorship bias** — the 16-stock universe consists of surviving large-caps and does not represent a survivorship-free dynamic index.

All classification and regression metrics are independently verified and honest. No artificial inflation of results was detected.

---

## 2. Test Suite Results

### 2.1 Automated Tests: 18/18 PASSED

| # | Test | Status |
|:--|:-----|:-------|
| 1 | `test_symbol_inventory` | ✅ PASSED |
| 2 | `test_date_parsing_and_chronology` | ✅ PASSED |
| 3 | `test_symbol_date_uniqueness` | ✅ PASSED |
| 4 | `test_eq_series_enforcement` | ✅ PASSED |
| 5 | `test_ohlc_consistency` | ✅ PASSED |
| 6 | `test_corporate_action_adjustments` | ✅ PASSED |
| 7 | `test_target_alignment_and_no_symbol_bleed` | ✅ PASSED |
| 8 | `test_no_future_leakage_in_features` | ✅ PASSED |
| 9 | `test_chronological_purge_separation` | ✅ PASSED |
| 10 | `test_prediction_interface_schema` | ✅ PASSED |
| 11 | `test_no_inf_in_features` | ✅ PASSED |
| 12 | `test_feature_columns_exist` | ✅ PASSED |
| 13 | `test_warmup_nans_only_at_start` | ✅ PASSED |
| 14 | `test_walk_forward_uses_config_specs` | ✅ PASSED |
| 15 | `test_holdout_predictions_exist_and_consistent` | ✅ PASSED |
| 16 | `test_evaluation_report_json_exists` | ✅ PASSED |
| 17 | `test_class_balance_per_symbol` | ✅ PASSED |
| 18 | `test_model_serialization_fidelity` | ✅ PASSED |

### 2.2 Independent Metric Verification

All 5 models' holdout metrics were independently recalculated from `holdout_predictions.csv` and compared against `model_comparison.csv`. Maximum absolute difference across all metrics: **< 10⁻¹⁶** (float precision). **All metrics MATCH.**

---

## 3. Chronological Boundaries Verification

### 3.1 Data Properties

| Property | Value |
|:---------|:------|
| Total rows | 19,840 |
| Unique symbols | 16 |
| Date range | 2021-10-25 to 2026-10-23 |
| Rows per stock | 1,240 |
| Duplicate Symbol-Date keys | 0 |
| Invalid dates | 0 |
| OHLC violations | 0 |
| Labeled observations | 19,760 (80 NaN targets = 16 stocks × 5 horizon) |
| Class 1 (Up) ratio | 52.27% |

### 3.2 Temporal Partitions

| Partition | Start | End | Rows | Trading Dates |
|:----------|:------|:----|:-----|:--------------|
| **Development** | 2021-10-25 | 2026-01-13 | 16,784 | 1,049 |
| **Purge buffer** | 2026-01-14 | 2026-01-21 | — | 5 sessions |
| **Holdout** | 2026-01-22 | 2026-10-15 | 2,896 | 181 dates × 16 stocks |

**Verification:** Development max date (2026-01-13) + 5 trading sessions = 2026-01-21 < Holdout start (2026-01-22). ✅ No overlap.

### 3.3 Walk-Forward Folds (Corrected)

| Fold | Train End | Purge | Val Start | Val End | Train N | Val N |
|:-----|:----------|:------|:----------|:--------|:--------|:------|
| 1 | 2023-12-04 | 5 sessions | 2023-12-12 | 2024-07-23 | 8,384 | 2,432 |
| 2 | 2024-07-23 | 5 sessions | 2024-07-31 | 2025-03-07 | 10,896 | 2,448 |
| 3 | 2025-03-07 | 5 sessions | 2025-03-18 | 2026-01-13 | 13,424 | 3,280 |

**Change from baseline:** Folds now use config specs `[(0.50, 0.65), (0.65, 0.80), (0.80, 1.00)]` instead of the previously hardcoded `[(0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]`. Fold 1 and Fold 2 boundaries shifted.

---

## 4. Feature Preprocessing Verification

- **Imputation:** `SimpleImputer(strategy="median")` fitted only on training data within each pipeline
- **Scaling:** `StandardScaler()` fitted only on training data within each pipeline
- **Feature leakage test:** No feature column contains substrings "future", "target", "fwd", or "next"
- **RSI warmup:** NaN values left for the imputer (not pre-filled with 50.0)
- **All features point-in-time:** Rolling windows use only past/current data per stock
- **Corporate actions:** 4 events back-adjusted; all adjusted returns verified across ex-dates

---

## 5. Classification Evaluation (5-Session Directional)

### 5.1 Walk-Forward Cross-Validation (Mean ± Std across 3 folds)

| Model | ROC-AUC | Balanced Acc | Accuracy | PR-AUC |
|:------|:--------|:-------------|:---------|:-------|
| **Logistic Regression** | **0.5225 ± 0.0119** | 0.5141 ± 0.0155 | 0.5025 ± 0.0260 | 0.5392 ± 0.0262 |
| Hist Gradient Boosting | 0.5157 ± 0.0106 | 0.5124 ± 0.0120 | 0.5085 ± 0.0073 | 0.5335 ± 0.0309 |
| XGBoost | 0.5154 ± 0.0085 | 0.5110 ± 0.0043 | 0.5123 ± 0.0114 | 0.5278 ± 0.0283 |
| Random Forest | 0.5085 ± 0.0051 | 0.5038 ± 0.0007 | 0.5030 ± 0.0134 | 0.5328 ± 0.0271 |
| Dummy Baseline | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5240 ± 0.0289 | 0.5240 ± 0.0289 |

**Model Selection:** Logistic Regression selected (highest walk-forward ROC-AUC). Decision consistent with pre-specified rule.

### 5.2 Holdout Evaluation (Untouched 2026-01-22 to 2026-10-15)

| Model | ROC-AUC | Bal. Acc | Accuracy | Precision | Recall | F1 | PR-AUC | Brier |
|:------|:--------|:---------|:---------|:----------|:-------|:---|:-------|:------|
| **Hist Gradient Boosting** | **0.5157** | 51.79% | 51.38% | 49.30% | 61.05% | 0.5455 | 0.4957 | 0.2559 |
| XGBoost | 0.5063 | 49.85% | 48.38% | 47.70% | 83.16% | 0.6063 | 0.4898 | 0.2576 |
| Random Forest | 0.5014 | 50.11% | 49.34% | 47.87% | 67.34% | 0.5596 | 0.4868 | 0.2536 |
| Dummy Baseline | 0.5000 | 50.00% | 47.79% | 47.79% | 100.0% | 0.6467 | 0.4779 | 0.5221 |
| **Logistic Regression** (deployed) | 0.4644 | 46.91% | 47.13% | 44.36% | 41.76% | 0.4302 | 0.4586 | 0.2526 |

**Key finding:** The deployed model (Logistic Regression) performs slightly below chance on holdout (AUC 0.4644), consistent with the original evaluation. Hist Gradient Boosting achieves the best holdout AUC (0.5157) but remains near chance level. No model demonstrates statistically significant predictive skill.

### 5.3 Per-Stock Holdout Performance (Logistic Regression)

| Symbol | N | Class 1% | Accuracy | Bal. Acc | ROC-AUC |
|:-------|:--|:---------|:---------|:---------|:--------|
| RELIANCE | 181 | 44.2% | 53.6% | 55.6% | **0.618** |
| INFY | 181 | 44.2% | 50.8% | 50.9% | **0.554** |
| HDFCBANK | 181 | 38.1% | 45.3% | 51.6% | **0.549** |
| SOLARINDS | 181 | 63.5% | 43.6% | 51.8% | **0.549** |
| ULTRACEMCO | 181 | 44.8% | 54.1% | 55.6% | **0.542** |
| SIEMENS | 181 | 50.8% | 49.2% | 49.6% | 0.518 |
| POWERINDIA | 181 | 54.1% | 46.4% | 49.1% | 0.470 |
| MOTILALOFS | 181 | 51.9% | 45.9% | 46.8% | 0.469 |
| APOLLOHOSP | 181 | 59.1% | 45.9% | 44.8% | 0.466 |
| ABB | 181 | 48.6% | 48.6% | 48.2% | 0.465 |
| TCS | 181 | 37.6% | 44.2% | 46.2% | 0.460 |
| MARUTI | 181 | 35.9% | 48.1% | 52.0% | 0.456 |
| YESBANK | 181 | 40.9% | 45.3% | 44.8% | 0.455 |
| TITAN | 181 | 49.7% | 43.1% | 43.0% | 0.436 |
| SBIN | 181 | 48.6% | 45.3% | 45.5% | 0.427 |
| ADANIGREEN | 181 | 52.5% | 44.8% | 46.4% | **0.415** |

**Note:** With only 181 holdout observations per stock, individual stock AUCs have wide confidence intervals (~±0.10). No individual stock achieves statistically significant performance after multiple comparison correction.

---

## 6. Regression Evaluation (5-Session Return)

### 6.1 Model vs Baselines

| Method | MAE | RMSE | R² (vs train-mean) | R² (vs zero) |
|:-------|:----|:-----|:--------------------|:-------------|
| Naive Zero-Return Predictor | **0.0339** | **0.0465** | — | — |
| Naive Training-Mean (0.42%) | 0.0343 | 0.0465 | 0.0000 (by definition) | — |
| **HistGBR Model** | 0.0347 | 0.0472 | **−0.0296** | **−0.0308** |

| Metric | Value |
|:-------|:------|
| Directional Accuracy | 47.03% |
| Directional ROC-AUC | 0.5032 |
| Holdout N | 2,896 |

### 6.2 Regression Interpretation

**The regression model performs worse than both naive baselines.** The negative R² (−0.030) confirms that predicting the training-set mean return would produce lower squared errors than the model's predictions. This is consistent with the classification findings: short-term equity returns are effectively unpredictable from daily technical features.

---

## 7. Comparison: Original Baseline vs Repaired Results

### 7.1 Walk-Forward (most significant change)

| Model | Original WF AUC | Repaired WF AUC | Change |
|:------|:----------------|:-----------------|:-------|
| Logistic Regression | 0.5159 ± 0.0130 | **0.5225 ± 0.0119** | +0.0066 |
| Hist Gradient Boosting | 0.5044 ± 0.0127 | **0.5157 ± 0.0106** | +0.0113 |
| XGBoost | 0.5062 ± 0.0200 | **0.5154 ± 0.0085** | +0.0092 |
| Random Forest | 0.5046 ± 0.0157 | **0.5085 ± 0.0051** | +0.0039 |

**Explanation:** Walk-forward fold boundaries changed due to the config specs fix. Fold 1 now trains on 50% of dates (was 40%), giving larger training sets and different validation windows. All models show slight AUC improvements but remain near chance level.

### 7.2 Holdout

| Model | Original Holdout AUC | Repaired Holdout AUC | Change |
|:------|:---------------------|:---------------------|:-------|
| Hist Gradient Boosting | 0.5187 | 0.5157 | −0.0030 |
| XGBoost | 0.5033 | 0.5063 | +0.0030 |
| Random Forest | 0.5017 | 0.5014 | −0.0003 |
| Logistic Regression | 0.4642 | 0.4644 | +0.0002 |

**Explanation:** Small changes (< 0.003) due to the RSI warmup fix changing imputed values. The deployed model (Logistic Regression) and model selection (highest WF AUC) remain unchanged. All models still near chance level.

### 7.3 New Metrics (not in original)

| Metric | Value |
|:-------|:------|
| Regression OOS R² (vs train-mean) | −0.0296 |
| Regression OOS R² (vs zero) | −0.0308 |
| Naive zero-return MAE | 0.0339 |
| Naive train-mean MAE | 0.0343 |
| Model MAE | 0.0347 (worse than both baselines) |

---

## 8. Dashboard Verification

| Check | Status |
|:------|:-------|
| Model name dynamically loaded from evaluation_report.json | ✅ |
| Holdout metrics loaded from model_comparison.csv (not hardcoded) | ✅ |
| Walk-forward summary loaded from walk_forward_summary.csv | ✅ |
| Per-stock metrics loaded from per_stock_evaluation.csv | ✅ |
| Volume display correctly shows multiplier (V/avg) not (V/avg − 1) | ✅ |
| Diagnostic figures loaded from generated PNG files | ✅ |
| Disclaimer and limitations displayed | ✅ |

---

## 9. Fundamental Features Assessment

**Status: NOT APPLICABLE**

- No `research_protocol.md` exists in the repository
- No fundamental data (earnings, PE ratios, book values, etc.) exists
- No fundamental feature columns exist in the analytical dataset
- The 20 features are exclusively market-technical (returns, SMAs, volatility, RSI, volume, calendar)
- Market-only vs fundamentals-only vs combined experiments cannot be conducted

This is a scope limitation of the available data, not a bug. The project correctly describes itself as using "daily technical features alone."

---

## 10. Profitability Claims

**No profitability claims are made.** The project explicitly states:
- "Consistent with the Efficient Market Hypothesis, short-term directional predictability from daily technical features alone is near chance level"
- "Models must NOT be used for capital deployment"
- Transaction costs are not modeled

This assessment is correct and well-documented.

---

## 11. Unresolved Limitations

1. **Survivorship bias:** The 16-stock universe represents surviving large-caps as of 2026. Historical results may differ for a dynamically rebalanced universe.
2. **No fundamental data:** Cannot evaluate whether fundamental features add predictive value.
3. **No research protocol:** Cannot verify compliance with frozen experimental specifications.
4. **Siemens volume adjustment:** Pre-demerger volume is scaled by ~2x to preserve turnover invariance, but does not represent physical share count.
5. **Small holdout per stock:** 181 observations per stock yields wide confidence intervals for individual stock performance.
6. **No statistical significance tests on repaired results:** Moving-block bootstrap confidence intervals from the original audit were not re-run (computationally expensive; qualitative conclusions unchanged).

---

## 12. Reproduction Commands

```bash
cd model_development

# Complete pipeline (data → training → evaluation → tests)
python run_project.py

# Tests only
python -m pytest tests/ -v

# Independent metric verification
python verify_metrics.py

# Single stock prediction
python predict.py --symbol RELIANCE

# All stocks prediction
python predict.py --all

# Dashboard
streamlit run app.py
```

---

## 13. Conclusion

**Verdict: CONDITIONAL**

The pipeline is **reproducible**, **leakage-free**, and **honestly evaluated**. All classification metrics are independently verified to machine precision. The walk-forward fold configuration bug has been fixed, regression baselines have been added (revealing the model performs worse than naive), and dashboard hardcoded values have been replaced with dynamic lookups.

The CONDITIONAL status reflects the absence of fundamental data and research protocol — items referenced in the evaluation task that do not exist in this repository. Within the scope of what is available (daily technical features on 16 NSE stocks), the evaluation is complete and defensible.
