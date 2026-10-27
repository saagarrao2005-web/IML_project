# Independent Quantitative ML Audit Report

**Audited Branch:** `prepare/training-pipeline`  
**Audited Artifacts:** [`experiment_results.md`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/reports/experiment_results.md), [`development_classification_predictions.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/predictions/development_classification_predictions.csv), [`development_regression_predictions.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/predictions/development_regression_predictions.csv), [`walk_forward_classification_results.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/reports/walk_forward_classification_results.csv), [`walk_forward_regression_results.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/reports/walk_forward_regression_results.csv), [`training_pipeline.py`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/training_pipeline.py)  
**Audit Date:** 2026-10-27  
**Role:** Independent Quantitative ML Auditor & Financial Econometrician  
**Final Holdout Status:** **STRICTLY SEQUESTERED & UNTOUCHED (Zero Scoring / Zero Tuning)**

---

## 1. Executive Audit Summary

An independent, rigorous quantitative audit was executed against the actual codebase, experimental pipelines, prediction records, and serialized model artifacts. No claims from `experiment_results.md` were accepted without algorithmic reproduction directly from the underlying data and code.

### Audit Verdict Summary Table

| Audit Check | Scope | Verdict | Primary Evidence |
| :--- | :--- | :---: | :--- |
| **Check 1** | Metric Recalculation from Raw Predictions | **PASS** | 100% mathematical reconciliation of fold metrics across both tasks from raw prediction CSVs. Discrepancies between fold-averaged vs. pooled metrics analyzed. |
| **Check 2** | Target Formulation & Price Alignment | **PASS** | Verified Task A ($H=5$) direction and Task B ($H=63$) continuous returns. Zero target bleed into feature matrices. |
| **Check 3** | Preprocessing & Pipeline Containment | **PASS** | Imputers and scalers strictly encapsulated within scikit-learn `Pipeline`; fitted solely on training folds. |
| **Check 4** | Temporal Purge Buffers & Label Overlap | **PASS** | Formal proof: $\max(\text{train target date}) < \min(\text{val date})$ across all folds, stocks, and tasks. Zero temporal bleed. |
| **Check 5** | Majority-Class Baseline & Aggregation | **PASS** | Baseline dynamically predicted training-fold majority class (Class 1 across all 3 folds). |
| **Check 6** | Dataset Integrity & Model Serialization | **PASS** | 0 duplicate keys, 0 NaNs, 16 symbols fully balanced (510 val rows in Task A, 311 in Task B). Saved models match configs. |
| **Check 7** | Econometric $R^2$ & EMH Claims Review | **PASS** | Out-of-sample $R^2$ formulas verified. Overstated causal claims of "proving EMH" replaced with bounded econometric findings. |
| **Check 8** | Automated Regression Test Suite | **PASS** | Added 4 new audit tests to `tests/test_pipeline.py`. **27 / 27 pytest tests passing**. |

### Final Audit Decision
### **`VALIDATION RESULTS TRUSTWORTHY`**
*(All underlying experimental predictions, fold metrics, and models are fully verified; minor markdown table metadata discrepancies were identified and corrected).*

---

## 2. Check 1: Metric Recalculation from Saved Out-of-Sample Predictions

Every metric reported in `experiment_results.md` was independently recomputed directly from the row-level predictions stored in:
- [`outputs/predictions/development_classification_predictions.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/predictions/development_classification_predictions.csv) (8,160 rows)
- [`outputs/predictions/development_regression_predictions.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/predictions/development_regression_predictions.csv) (4,976 rows)

### 2.1 Task A Classification: Fold Reconciliation

For each of the 3 expanding folds and 5 models, metrics were recomputed from `actual`, `pred`, and `prob` columns.

```text
Fold 1 (N = 2,432):
  Majority_Class_Baseline: Acc=0.5761, BalAcc=0.5000, AUC=0.5000, F1=0.7310, CM=[0, 1031, 0, 1401]  -> EXACT MATCH
  Logistic_Regression:     Acc=0.4716, BalAcc=0.5026, AUC=0.5183, F1=0.3947, CM=[728, 303, 982, 419]  -> EXACT MATCH
  Random_Forest:           Acc=0.4918, BalAcc=0.5043, AUC=0.5139, F1=0.4888, CM=[605, 426, 810, 591]  -> EXACT MATCH
  Hist_Gradient_Boosting:  Acc=0.5004, BalAcc=0.5164, AUC=0.5205, F1=0.4867, CM=[641, 390, 825, 576]  -> EXACT MATCH
  XGBoost:                 Acc=0.5169, BalAcc=0.5098, AUC=0.5209, F1=0.5701, CM=[478, 553, 622, 779]  -> EXACT MATCH

Fold 2 (N = 2,448):
  Majority_Class_Baseline: Acc=0.4624, BalAcc=0.5000, AUC=0.5000, F1=0.6324, CM=[0, 1316, 0, 1132]  -> EXACT MATCH
  Logistic_Regression:     Acc=0.5208, BalAcc=0.5328, AUC=0.5359, F1=0.5717, CM=[492, 824, 349, 783]  -> EXACT MATCH
  Random_Forest:           Acc=0.4996, BalAcc=0.5031, AUC=0.5078, F1=0.5038, CM=[601, 715, 510, 622]  -> EXACT MATCH
  Hist_Gradient_Boosting:  Acc=0.5192, BalAcc=0.5224, AUC=0.5231, F1=0.5210, CM=[631, 685, 492, 640]  -> EXACT MATCH
  XGBoost:                 Acc=0.4886, BalAcc=0.5149, AUC=0.5198, F1=0.6100, CM=[217, 1099, 153, 979] -> EXACT MATCH

Fold 3 (N = 3,280):
  Majority_Class_Baseline: Acc=0.5335, BalAcc=0.5000, AUC=0.5000, F1=0.6958, CM=[0, 1530, 0, 1750]  -> EXACT MATCH
  Logistic_Regression:     Acc=0.5149, BalAcc=0.5069, AUC=0.5133, F1=0.5794, CM=[593, 937, 654, 1096] -> EXACT MATCH
  Random_Forest:           Acc=0.5177, BalAcc=0.5040, AUC=0.5037, F1=0.6102, CM=[460, 1070, 512, 1238]-> EXACT MATCH
  Hist_Gradient_Boosting:  Acc=0.5058, BalAcc=0.4982, AUC=0.5035, F1=0.5688, CM=[590, 940, 681, 1069] -> EXACT MATCH
  XGBoost:                 Acc=0.5314, BalAcc=0.5084, AUC=0.5056, F1=0.6596, CM=[254, 1276, 261, 1489]-> EXACT MATCH
```

### 2.2 Reconciling Fold-Averaged vs. Pooled Metrics
The audit revealed an important technical distinction:
- **Fold-Averaged Summary (Report Table 3.1):** Reports the unweighted mean and standard deviation across the 3 folds ($\bar{m} = \frac{1}{3}\sum_{k=1}^3 m_k$).
- **Pooled Metrics (Concatenated across all 8,160 rows):** Evaluates all out-of-sample predictions together.

| Model | Fold-Averaged ROC-AUC | Pooled ROC-AUC | Fold-Averaged Bal Acc | Pooled Bal Acc | Fold-Averaged F1 | Pooled F1 | Cumulative Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.5000 ± 0.0000 | 0.5000 | 50.00% | 50.00% | 0.6864 | 0.6884 | [0, 3877, 0, 4283] |
| **Logistic Regression** | **0.5225 ± 0.0119** | **0.5060** | **51.41%** | **50.21%** | 0.5153 | 0.5316 | [1813, 2064, 1985, 2298] |
| **Random Forest** | 0.5085 ± 0.0051 | 0.5001 | 50.38% | 50.10% | 0.5343 | 0.5480 | [1666, 2211, 1832, 2451] |
| **HistGradientBoosting** | 0.5157 ± 0.0106 | 0.5093 | 51.24% | 50.69% | 0.5255 | 0.5324 | [1862, 2015, 1998, 2285] |
| **XGBoost** | 0.5154 ± 0.0085 | 0.5028 | 51.10% | 50.14% | 0.6132 | 0.6210 | [949, 2928, 1036, 3247] |

**Explanation of Discrepancy:**  
Pooled ROC-AUC (0.5060) is lower than the fold-averaged mean ROC-AUC (0.5225) because the underlying market regime drifted between folds (Fold 1 had 57.6% upward drift, Fold 2 had 46.2%, Fold 3 had 53.4%). When probabilities generated with uncalibrated intercept shifts are pooled across non-stationary market regimes, ranking consistency degrades slightly. Fold-averaged reporting is the standard econometric practice for walk-forward validation and is correctly reflected in the report summary.

### 2.3 Task B Regression: Fold Reconciliation

```text
Fold 1 (N = 1,392, tr_mean = 0.050893):
  Zero Return Baseline:   MAE=0.1689, RMSE=0.2580, R2(Mean)=-0.2010, DirAcc=0.4579, Rank IC=0.0000  -> EXACT MATCH
  Training Mean Baseline: MAE=0.1552, RMSE=0.2355, R2(Mean)=0.0000,  DirAcc=0.5421, Rank IC=0.0000  -> EXACT MATCH
  Ridge Regression:       MAE=0.1599, RMSE=0.2434, R2(Mean)=-0.0689, DirAcc=0.5147, Rank IC=+0.0018 -> EXACT MATCH
  Random Forest Regressor:MAE=0.1583, RMSE=0.2413, R2(Mean)=-0.0500, DirAcc=0.5169, Rank IC=+0.0535 -> EXACT MATCH
  HistGradientBoosting:   MAE=0.1595, RMSE=0.2474, R2(Mean)=-0.1040, DirAcc=0.5239, Rank IC=+0.0251 -> EXACT MATCH

Fold 2 (N = 1,392, tr_mean = 0.075689):
  Zero Return Baseline:   MAE=0.1248, RMSE=0.1662, R2(Mean)=+0.4140, DirAcc=0.5267, Rank IC=0.0000  -> EXACT MATCH
  Training Mean Baseline: MAE=0.1792, RMSE=0.2170, R2(Mean)=0.0000,  DirAcc=0.4733, Rank IC=0.0000  -> EXACT MATCH
  Ridge Regression:       MAE=0.1837, RMSE=0.2246, R2(Mean)=-0.0711, DirAcc=0.4375, Rank IC=-0.0934 -> EXACT MATCH
  Random Forest Regressor:MAE=0.1782, RMSE=0.2214, R2(Mean)=-0.0403, DirAcc=0.4775, Rank IC=-0.1337 -> EXACT MATCH
  HistGradientBoosting:   MAE=0.1765, RMSE=0.2210, R2(Mean)=-0.0369, DirAcc=0.4789, Rank IC=-0.0318 -> EXACT MATCH

Fold 3 (N = 2,192, tr_mean = 0.053269):
  Zero Return Baseline:   MAE=0.0830, RMSE=0.1092, R2(Mean)=+0.0292, DirAcc=0.4962, Rank IC=0.0000  -> EXACT MATCH
  Training Mean Baseline: MAE=0.0873, RMSE=0.1108, R2(Mean)=0.0000,  DirAcc=0.5038, Rank IC=0.0000  -> EXACT MATCH
  Ridge Regression:       MAE=0.0883, RMSE=0.1108, R2(Mean)=-0.0002, DirAcc=0.5160, Rank IC=-0.0726 -> EXACT MATCH
  Random Forest Regressor:MAE=0.0879, RMSE=0.1111, R2(Mean)=-0.0057, DirAcc=0.5117, Rank IC=+0.0462 -> EXACT MATCH
  HistGradientBoosting:   MAE=0.0894, RMSE=0.1123, R2(Mean)=-0.0274, DirAcc=0.5470, Rank IC=+0.0772 -> EXACT MATCH
```

---

## 3. Check 2 & Check 3: Target Alignment, Leakage & Preprocessing Isolation

### 3.1 Target Alignment Verification
- **Task A Target:** Built via `grouped["adj_close"].shift(-5)`. Labeled rows = 19,760 with exactly $16 \times 5 = 80$ terminal NaNs.
- **Task B Target:** Built via `grouped["adj_close"].shift(-63)`. Labeled rows = 18,832 with exactly $16 \times 63 = 1,008$ terminal NaNs.
- **Target Bleed Check:** Feature matrix columns strictly match `MARKET_FEATURES` in `config.py`. All target and future price series (`future_adj_close_5d`, `future_return_5d`, `target_up_5d`, `future_adj_close_63d`, `future_return_63d`) are quarantined in `METADATA_COLUMNS`.

### 3.2 Preprocessing Containment Verification
- In [`training_pipeline.py#L110-L208`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/training_pipeline.py#L110-L208), every model is wrapped in a `Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", ...)])`.
- `pipe.fit(X_tr, y_tr)` is called exclusively on the training fold slice.
- `pipe.predict(X_vl)` uses the imputer statistics and scaler parameters computed from `X_tr`. Zero validation statistics leak into the transforms.
- In `compute_features()`, all moving averages and volatility windows use `.rolling(window, min_periods=window)` and `.pct_change(lag)`, which are strictly backward-looking. Wilder's RSI warmup NaNs are preserved for the fold imputer to handle inside the training fold.

---

## 4. Check 4: Walk-Forward Split Boundaries, Purge Logic & Multi-Stock Overlap

### 4.1 Temporal Purge Buffer Mechanics
In time series forecasting with multi-step forward targets, if observation $t$ predicts return over $[t, t+H]$, the label for training observation $T_{\text{train\_end}}$ is only realized at $T_{\text{train\_end}} + H$. If validation starts at $T_{\text{train\_end}} + 1$, validation observations use historical prices overlapping the training label's forward horizon.

The pipeline prevents this by enforcing an explicit purge buffer $B \ge H$:
$$\text{val\_start\_idx} = \text{tr\_end\_idx} + B$$

### 4.2 Mathematical Proof of Zero Overlap Across All Folds

| Task | Fold | Last Train Date ($t_{\text{tr}}$) | Train Target Realization ($t_{\text{tr}} + H$) | First Val Date ($t_{\text{val}}$) | Temporal Buffer Gap | Overlap? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task A** ($H=5, B=5$) | **Fold 1** | 2023-12-04 (idx 523) | **2023-12-11 (idx 528)** | **2023-12-12 (idx 529)** | 6 dates | **FALSE (No Overlap)** |
| **Task A** ($H=5, B=5$) | **Fold 2** | 2024-07-23 (idx 680) | **2024-07-30 (idx 685)** | **2024-07-31 (idx 686)** | 6 dates | **FALSE (No Overlap)** |
| **Task A** ($H=5, B=5$) | **Fold 3** | 2025-03-07 (idx 838) | **2025-03-17 (idx 843)** | **2025-03-18 (idx 844)** | 6 dates | **FALSE (No Overlap)** |
| **Task A** (Dev $\to$ Holdout) | Holdout | 2026-01-13 (idx 1048) | **2026-01-20 (idx 1053)** | **2026-01-22 (idx 1054)** | 6 dates | **FALSE (No Overlap)** |
| **Task B** ($H=63, B=63$) | **Fold 1** | 2023-10-30 (idx 499) | **2024-01-30 (idx 562)** | **2024-01-31 (idx 563)** | 64 dates | **FALSE (No Overlap)** |
| **Task B** ($H=63, B=63$) | **Fold 2** | 2024-06-06 (idx 649) | **2024-09-06 (idx 712)** | **2024-09-09 (idx 713)** | 64 dates | **FALSE (No Overlap)** |
| **Task B** ($H=63, B=63$) | **Fold 3** | 2025-01-13 (idx 799) | **2025-04-16 (idx 862)** | **2025-04-17 (idx 863)** | 64 dates | **FALSE (No Overlap)** |
| **Task B** (Dev $\to$ Holdout) | Holdout | 2025-11-03 (idx 999) | **2026-02-03 (idx 1062)** | **2026-02-04 (idx 1063)** | 64 dates | **FALSE (No Overlap)** |

### 4.3 Multi-Stock Shared Calendar Handling
Because all 16 equities trade concurrently on the National Stock Exchange under identical exchange trading schedules, calendar dates are uniformly indexed across the panel. An exhaustive cross-sectional check verified that:
$$\max_{s \in \text{stocks}} (\text{target realization date of } s \text{ in train}) < \min_{s \in \text{stocks}} (\text{date of } s \text{ in val})$$
Zero cross-stock or cross-asset forward label leakage exists.

---

## 5. Check 5 & Check 6: Majority Baseline & Dataset Integrity

### 5.1 Majority-Class Baseline Verification
- In each training fold, `DummyClassifier(strategy="most_frequent")` correctly identified the mode of training labels:
  - Fold 1: Class 1 = 4,463 vs Class 0 = 3,921 (Mode = 1) $\to$ predicted 1.
  - Fold 2: Class 1 = 5,924 vs Class 0 = 4,972 (Mode = 1) $\to$ predicted 1.
  - Fold 3: Class 1 = 7,098 vs Class 0 = 6,326 (Mode = 1) $\to$ predicted 1.
- No static constant or fallback data was hardcoded.

### 5.2 Artifact Integrity & Completeness
- **Uniqueness:** 0 duplicate `(fold, Date, Symbol)` keys in both classification and regression prediction files.
- **Completeness:** 0 missing values or NaNs in predictions or probability estimates.
- **Bounds:** All predicted probabilities in Task A fall strictly within $[0.0672, 0.8898] \subset [0, 1]$.
- **Balance:** Every stock has exactly 510 out-of-sample predictions in Task A ($510 \times 16 = 8,160$) and exactly 311 out-of-sample predictions in Task B ($311 \times 16 = 4,976$).
- **Model Deserialization:** Serialized `.joblib` files in `outputs/models/` contain complete, uncorrupted pipelines fitted on full development datasets (16,784 rows for Task A; 16,000 rows for Task B).
- **Holdout Gate:** Both `experiment_config_classification.json` and `experiment_config_regression.json` explicitly record `"evaluated": false` for the final holdout period.

---

## 6. Check 7: Review of Regression $R^2$ Formulas & EMH Claims

### 6.1 Audit of $R^2$ Formulas
The pipeline computes two distinct out-of-sample coefficients of determination:
1. **$R^2$ vs. Training Mean:**
   $$R^2_{\text{vs mean}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y}_{\text{train}})^2}$$
2. **$R^2$ vs. Zero Benchmark:**
   $$R^2_{\text{vs zero}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum y_i^2}$$

Both formulas were independently recomputed from `development_regression_predictions.csv` and matched `walk_forward_regression_results.csv` exactly.

### 6.2 Correction of Unsupported Economic Conclusions
The original draft of `experiment_results.md` stated that negative out-of-sample $R^2$ demonstrates the Efficient Market Hypothesis. 

**Auditor Correction:**  
Under the Fama Joint Hypothesis Problem, an empirical test cannot isolate market efficiency from the joint specification of the pricing model, feature space, and sample constraints. A negative out-of-sample $R^2$ across 3 simple models on 16 stocks over a 4-year sample does **not** demonstrate market efficiency; it merely demonstrates that **these 20 technical features and 3 model architectures failed to extract predictive alpha for 3-month continuous returns on this universe**. The text in `experiment_results.md` has been amended to reflect this rigorous econometric qualification.

---

## 7. Discrepancies Identified and Corrections Made

During the audit, a discrepancy was identified between the markdown text in `experiment_results.md` and the actual generated CSV artifacts:

### Summary of Discrepancies and Fixes

| Item | Severity | Original Markdown in `experiment_results.md` | Actual Verified Artifact Data | Resolution / Correction |
| :--- | :---: | :--- | :--- | :--- |
| **Task A Fold 1 & 2 Train Rows** | **LOW** | Fold 1: 8,392 / Fold 2: 10,904 | Fold 1: **8,384** / Fold 2: **10,896** | Updated table in `experiment_results.md` to match `walk_forward_classification_results.csv`. |
| **Task B Fold Val Rows** | **MEDIUM** | Fold 1: 1,424 / Fold 2: 1,424 / Fold 3: 2,128 | Fold 1: **1,392** / Fold 2: **1,392** / Fold 3: **2,192** | Updated table in `experiment_results.md` to match `walk_forward_regression_results.csv`. |
| **Fold Date Intervals in Markdown** | **LOW** | Approximate calendar day bounds from draft notes | Exact trading session date intervals | Updated date interval strings in `experiment_results.md` to reflect exact trading dates. |
| **Overstated EMH Wording** | **LOW** | Claimed negative $R^2$ "demonstrates EMH" | Joint hypothesis failure of technical features | Replaced with qualified econometric limitation in `experiment_results.md`. |

---

## 8. Check 8: Automated Regression Test Suite

Four new automated test functions were added to [`model_development/tests/test_pipeline.py`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/tests/test_pipeline.py):
1. `test_audit_task_a_recalculation_reconciliation`: Confirms that all classification metrics recomputed from raw CSVs match saved results.
2. `test_audit_task_b_recalculation_reconciliation`: Confirms that all regression metrics recomputed from raw CSVs match saved results.
3. `test_audit_no_temporal_label_overlap_in_walk_forward`: Programmatically asserts $\max(\text{train target date}) < \min(\text{val date})$ across all folds.
4. `test_audit_artifact_completeness_and_holdout_protection`: Confirms artifact row counts, key uniqueness, non-degeneracy, and un-evaluated holdout status.

### Test Execution Output
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
rootdir: D:\B.Tech\Sem3\IML\Project\Stock_Prediction
plugins: anyio-4.13.0, cov-7.1.0
collected 27 items

model_development\tests\test_pipeline.py ...........................     [100%]

============================= 27 passed in 7.00s ==============================
```

---

## 9. Reproduction Commands

To independently reproduce the complete audit verification suite and data checks without touching the holdout:

```powershell
# 1. Run the entire test suite including the 4 new audit tests
pytest model_development/tests/test_pipeline.py -v

# 2. Run data and provenance validator
python model_development/validate_data.py

# 3. Verify clean execution of training pipeline smoke test
python model_development/training_pipeline.py --smoke-test
```

---

## 10. Final Auditor Verdict

### **`VALIDATION RESULTS TRUSTWORTHY`**

The development validation results reported for Task A and Task B on branch `prepare/training-pipeline` are genuine, mathematically verifiable, free from data leakage, and strictly separated from the sequestered holdout partition. All identified metadata discrepancies in report tables have been fully reconciled.
