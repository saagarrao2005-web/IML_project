# Formal Walk-Forward Development Experiment Results

**Project:** Indian NSE Stock Prediction using Market Data and Fundamental Data  
**Branch:** `prepare/training-pipeline`  
**Execution Date:** 2026-10-27  
**Status:** REAL-DATA DEVELOPMENT EXPERIMENTS COMPLETE (Holdout Partition Strictly Sequestered)

---

## 1. Executive Summary

This report presents the empirical results of the first formal, leakage-free machine learning experiments conducted on the actual Indian National Stock Exchange (NSE) dataset (16 equities, 1,240 trading sessions per equity, spanning 2021-10-25 to 2026-10-23). 

All experiments were executed using expanding-window walk-forward cross-validation on the development partition with strict embargo/purge buffers to prevent look-ahead bias and forward-label bleed. In strict accordance with the quantitative research protocol, the sequestered holdout partition has **not** been touched or evaluated.

### Core Quantitative Findings

1. **Task A (5-Trading-Session Directional Classification):**
   - **Baseline:** Majority-class baseline achieved an ROC-AUC of 0.5000 and balanced accuracy of 50.00%.
   - **Winner:** **Logistic Regression** achieved the highest mean out-of-sample ROC-AUC of **0.5225 ± 0.0119** and balanced accuracy of **51.41% ± 1.63%**, outperforming Tree-based and Boosting ensembles (HistGradientBoosting: 0.5157, XGBoost: 0.5154, Random Forest: 0.5085).
   - **Market Signal:** A modest, statistically detectable technical signal exists at the 5-day horizon, where regularized linear representations generalize better than complex non-linear ensembles due to lower variance in high-noise regimes.

2. **Task B (63-Trading-Session Continuous-Return Regression):**
   - **Baseline:** The naive Zero-Return Baseline achieved the lowest MAE (**0.1256 ± 0.0430**) and positive out-of-sample $R^2$ (+0.0807 vs. training mean).
   - **ML Performance:** All complex ML regressors (Ridge Regression, Random Forest Regressor, HistGradientBoosting Regressor) exhibited **negative out-of-sample $R^2$** relative to the historical training mean (Random Forest: -0.0320, Ridge: -0.0467, HistGBR: -0.0561).
   - **Empirical Reality:** Within this specific universe of 16 NSE equities and across this 4-year window, the 20 lagged technical features evaluated by these three regression architectures failed to generate predictive signal superior to naive zero or unconditional mean baselines in terms of out-of-sample $R^2$. While consistent with weak-form market efficiency under the joint hypothesis problem, this reflects a failure of this specific technical feature set and model family to extract 3-month continuous return alpha, rather than a universal proof of market efficiency.

3. **Fundamental Feature Track Status:**
   - **Status:** **BLOCKED**. Zero financial filings exist in the workspace. No synthetic or fabricated fundamental values were introduced.

---

## 2. Experimental Setup and Chronological Partitions

### Universe and Dataset
- **Equities (16):** ABB, ADANIGREEN, APOLLOHOSP, HDFCBANK, INFY, MARUTI, MOTILALOFS, POWERINDIA, RELIANCE, SBIN, SIEMENS, SOLARINDS, TCS, TITAN, ULTRACEMCO, YESBANK.
- **Total Dataset Size:** 19,840 rows (1,240 sessions per stock).
- **Date Range:** 2021-10-25 to 2026-10-23.
- **Engineered Technical Features (20):** `return_1d`, `return_2d`, `return_5d`, `return_10d`, `return_20d`, `close_vs_sma_5`, `close_vs_sma_10`, `close_vs_sma_20`, `close_vs_sma_50`, `volatility_10d`, `volatility_20d`, `daily_range_pct`, `open_close_return`, `volume_change_1d`, `volume_vs_avg20`, `trades_vs_avg20`, `delivery_pct`, `rsi_14`, `day_of_week`, `month`.

### Partition Specifications

| Task | Horizon ($H$) | Purge Buffer ($B$) | Total Labeled | Development Window | Development Rows | Sequestered Holdout Window | Holdout Rows | Holdout Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Task A** | 5 sessions | 5 sessions | 19,760 | 2021-10-25 to 2026-01-13 | 16,784 | 2026-01-22 to 2026-10-15 | 2,896 | **UNTOUCHED** |
| **Task B** | 63 sessions | 63 sessions | 18,832 | 2021-10-25 to 2025-11-03 | 16,000 | 2026-02-04 to 2026-07-23 | 1,824 | **UNTOUCHED** |

### Expanding-Window Walk-Forward Splits
Both tasks evaluate across 3 expanding folds on the development partition using specifications `[(0.50, 0.65), (0.65, 0.80), (0.80, 1.00)]`:
- **Fold 1:** Train on first 50% of dates; purge $B$ dates; evaluate on next 15% of dates.
- **Fold 2:** Train on first 65% of dates; purge $B$ dates; evaluate on next 15% of dates.
- **Fold 3:** Train on first 80% of dates; purge $B$ dates; evaluate on final 20% of development dates.

---

## 3. Task A: 5-Day Directional Classification Results

### Target Formulation
- Binary classification: $y_t = \mathbb{I}(P_{t+5} > P_t)$.
- Development class balance: 52.27% positive instances (modest upward equity market drift).

### Walk-Forward Summary Comparison Table

| Architecture | Mean ROC-AUC | Std ROC-AUC | Mean Bal Acc | Precision | Recall | F1 Score | Brier Score | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 0.5000 | ±0.0000 | 0.5000 | 0.5240 | 1.0000 | 0.6864 | 0.4760 | Benchmark |
| **Logistic Regression** | **0.5225** | **±0.0119** | **0.5141** | **0.5356** | 0.5390 | 0.5153 | **0.2517** | **WINNER** |
| **Random Forest** | 0.5085 | ±0.0051 | 0.5038 | 0.5276 | 0.5596 | 0.5343 | 0.2528 | |
| **HistGradientBoosting** | 0.5157 | ±0.0106 | 0.5124 | 0.5371 | 0.5291 | 0.5255 | 0.2569 | |
| **XGBoost** | 0.5154 | ±0.0085 | 0.5110 | 0.5315 | 0.7572 | 0.6132 | 0.2540 | |

### Fold-by-Fold Detailed Breakdown

| Fold | Model | Train / Val Dates | Train / Val Rows | ROC-AUC | Balanced Acc | Precision | Recall | F1 Score | Confusion Matrix [TN, FP, FN, TP] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Majority Class | 2021-10-25..2023-12-04 / 2023-12-12..2024-07-23 | 8,384 / 2,432 | 0.5000 | 0.5000 | 0.5761 | 1.0000 | 0.7310 | [0, 1031, 0, 1401] |
| **1** | Logistic Regression | 2021-10-25..2023-12-04 / 2023-12-12..2024-07-23 | 8,384 / 2,432 | 0.5183 | 0.5026 | 0.5803 | 0.2991 | 0.3947 | [728, 303, 982, 419] |
| **1** | Random Forest | 2021-10-25..2023-12-04 / 2023-12-12..2024-07-23 | 8,384 / 2,432 | 0.5139 | 0.5043 | 0.5811 | 0.4218 | 0.4888 | [605, 426, 810, 591] |
| **1** | HistGradientBoosting | 2021-10-25..2023-12-04 / 2023-12-12..2024-07-23 | 8,384 / 2,432 | 0.5205 | 0.5164 | 0.5963 | 0.4111 | 0.4867 | [641, 390, 825, 576] |
| **1** | XGBoost | 2021-10-25..2023-12-04 / 2023-12-12..2024-07-23 | 8,384 / 2,432 | 0.5209 | 0.5098 | 0.5848 | 0.5560 | 0.5701 | [478, 553, 622, 779] |
| **2** | Majority Class | 2021-10-25..2024-07-23 / 2024-07-31..2025-03-07 | 10,896 / 2,448 | 0.5000 | 0.5000 | 0.4624 | 1.0000 | 0.6324 | [0, 1316, 0, 1132] |
| **2** | Logistic Regression | 2021-10-25..2024-07-23 / 2024-07-31..2025-03-07 | 10,896 / 2,448 | 0.5359 | 0.5328 | 0.4872 | 0.6917 | 0.5717 | [492, 824, 349, 783] |
| **2** | Random Forest | 2021-10-25..2024-07-23 / 2024-07-31..2025-03-07 | 10,896 / 2,448 | 0.5078 | 0.5031 | 0.4652 | 0.5495 | 0.5038 | [601, 715, 510, 622] |
| **2** | HistGradientBoosting | 2021-10-25..2024-07-23 / 2024-07-31..2025-03-07 | 10,896 / 2,448 | 0.5231 | 0.5224 | 0.4830 | 0.5654 | 0.5210 | [631, 685, 492, 640] |
| **2** | XGBoost | 2021-10-25..2024-07-23 / 2024-07-31..2025-03-07 | 10,896 / 2,448 | 0.5198 | 0.5149 | 0.4711 | 0.8648 | 0.6100 | [217, 1099, 153, 979] |
| **3** | Majority Class | 2021-10-25..2025-03-07 / 2025-03-18..2026-01-13 | 13,424 / 3,280 | 0.5000 | 0.5000 | 0.5335 | 1.0000 | 0.6958 | [0, 1530, 0, 1750] |
| **3** | Logistic Regression | 2021-10-25..2025-03-07 / 2025-03-18..2026-01-13 | 13,424 / 3,280 | 0.5133 | 0.5069 | 0.5391 | 0.6263 | 0.5794 | [593, 937, 654, 1096] |
| **3** | Random Forest | 2021-10-25..2025-03-07 / 2025-03-18..2026-01-13 | 13,424 / 3,280 | 0.5037 | 0.5040 | 0.5364 | 0.7074 | 0.6102 | [460, 1070, 512, 1238] |
| **3** | HistGradientBoosting | 2021-10-25..2025-03-07 / 2025-03-18..2026-01-13 | 13,424 / 3,280 | 0.5035 | 0.4982 | 0.5321 | 0.6109 | 0.5688 | [590, 940, 681, 1069] |
| **3** | XGBoost | 2021-10-25..2025-03-07 / 2025-03-18..2026-01-13 | 13,424 / 3,280 | 0.5056 | 0.5084 | 0.5385 | 0.8509 | 0.6596 | [254, 1276, 261, 1489] |

| Fold | Model | Train / Val Rows | ROC-AUC | Balanced Acc | Precision | Recall | F1 Score | Confusion Matrix [TN, FP, FN, TP] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Majority Class | 8,392 / 2,432 | 0.5000 | 0.5000 | 0.5761 | 1.0000 | 0.7310 | [0, 1031, 0, 1401] |
| **1** | Logistic Regression | 8,392 / 2,432 | 0.5183 | 0.5026 | 0.5803 | 0.2991 | 0.3947 | [728, 303, 982, 419] |
| **1** | Random Forest | 8,392 / 2,432 | 0.5139 | 0.5043 | 0.5811 | 0.4218 | 0.4888 | [605, 426, 810, 591] |
| **1** | HistGradientBoosting | 8,392 / 2,432 | 0.5205 | 0.5164 | 0.5963 | 0.4111 | 0.4867 | [641, 390, 825, 576] |
| **1** | XGBoost | 8,392 / 2,432 | 0.5209 | 0.5098 | 0.5848 | 0.5560 | 0.5701 | [478, 553, 622, 779] |
| **2** | Majority Class | 10,904 / 2,448 | 0.5000 | 0.5000 | 0.4624 | 1.0000 | 0.6324 | [0, 1316, 0, 1132] |
| **2** | Logistic Regression | 10,904 / 2,448 | 0.5359 | 0.5328 | 0.4872 | 0.6917 | 0.5717 | [492, 824, 349, 783] |
| **2** | Random Forest | 10,904 / 2,448 | 0.5078 | 0.5031 | 0.4652 | 0.5495 | 0.5038 | [601, 715, 510, 622] |
| **2** | HistGradientBoosting | 10,904 / 2,448 | 0.5231 | 0.5224 | 0.4830 | 0.5654 | 0.5210 | [631, 685, 492, 640] |
| **2** | XGBoost | 10,904 / 2,448 | 0.5198 | 0.5149 | 0.4711 | 0.8648 | 0.6100 | [217, 1099, 153, 979] |
| **3** | Majority Class | 13,424 / 3,280 | 0.5000 | 0.5000 | 0.5335 | 1.0000 | 0.6958 | [0, 1530, 0, 1750] |
| **3** | Logistic Regression | 13,424 / 3,280 | 0.5133 | 0.5069 | 0.5391 | 0.6263 | 0.5794 | [593, 937, 654, 1096] |
| **3** | Random Forest | 13,424 / 3,280 | 0.5037 | 0.5040 | 0.5364 | 0.7074 | 0.6102 | [460, 1070, 512, 1238] |
| **3** | HistGradientBoosting | 13,424 / 3,280 | 0.5035 | 0.4982 | 0.5321 | 0.6109 | 0.5688 | [590, 940, 681, 1069] |
| **3** | XGBoost | 13,424 / 3,280 | 0.5056 | 0.5084 | 0.5385 | 0.8509 | 0.6596 | [254, 1276, 261, 1489] |

### Cumulative Confusion Matrix (8,160 Development Observations)
For **Logistic Regression**:
- **True Negatives (TN):** 1,813 (46.8% of actual down-periods correctly identified)
- **False Positives (FP):** 2,064
- **False Negatives (FN):** 1,985
- **True Positives (TP):** 2,298 (53.7% of actual up-periods correctly identified)

---

## 4. Task B: 63-Day Continuous-Return Regression Results

### Target Formulation
- Continuous return: $y_t = \frac{P_{t+63}}{P_t} - 1.0$.
- Purge buffer: 63 trading sessions (1 calendar quarter) between training end and validation start.

### Walk-Forward Summary Comparison Table

| Architecture | Mean MAE | Std MAE | Mean RMSE | Std RMSE | OOS $R^2$ (vs Mean) | OOS $R^2$ (vs Zero) | Dir Acc | Mean Rank IC | IC $t$-stat | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero Return Baseline** | **0.1256** | **±0.0430** | **0.1778** | **±0.0751** | **+0.0807** | **0.0000** | 0.4936 | 0.0000 | 0.00 | **Lowest Loss** |
| **Training Mean Baseline** | 0.1406 | ±0.0476 | 0.1878 | ±0.0673 | 0.0000 | -0.1897 | 0.5064 | 0.0000 | 0.00 | Naive Mean |
| **Random Forest Regressor** | 0.1415 | ±0.0474 | 0.1913 | ±0.0701 | -0.0320 | -0.2285 | 0.5021 | -0.0114 | -0.34 | Best ML Regressor |
| **HistGradientBoosting** | 0.1418 | ±0.0462 | 0.1936 | ±0.0716 | -0.0561 | -0.2489 | **0.5166** | **+0.0235** | **+1.29** | Positive Rank IC |
| **Ridge Regression** | 0.1440 | ±0.0496 | 0.1930 | ±0.0717 | -0.0467 | -0.2493 | 0.4894 | -0.0548 | -2.18 | Overregularized |

### Fold-by-Fold Detailed Breakdown

| Fold | Model | Train / Val Dates | Train / Val Rows | MAE | RMSE | OOS ^2$ (Mean) | Directional Acc | Mean Rank IC | Rank IC $-stat |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Zero Return Baseline | 2021-10-25..2023-10-30 / 2024-01-31..2024-06-06 | 8,000 / 1,392 | 0.1689 | 0.2580 | -0.2010 | 0.4579 | 0.0000 | 0.00 |
| **1** | Training Mean Baseline | 2021-10-25..2023-10-30 / 2024-01-31..2024-06-06 | 8,000 / 1,392 | 0.1552 | 0.2355 | 0.0000 | 0.5421 | 0.0000 | 0.00 |
| **1** | Ridge Regression | 2021-10-25..2023-10-30 / 2024-01-31..2024-06-06 | 8,000 / 1,392 | 0.1599 | 0.2434 | -0.0689 | 0.5147 | +0.0018 | +0.06 |
| **1** | Random Forest Regressor | 2021-10-25..2023-10-30 / 2024-01-31..2024-06-06 | 8,000 / 1,392 | 0.1583 | 0.2413 | -0.0500 | 0.5169 | +0.0535 | +2.16 |
| **1** | HistGradientBoosting | 2021-10-25..2023-10-30 / 2024-01-31..2024-06-06 | 8,000 / 1,392 | 0.1595 | 0.2474 | -0.1040 | 0.5239 | +0.0251 | +0.97 |
| **2** | Zero Return Baseline | 2021-10-25..2024-06-06 / 2024-09-09..2025-01-13 | 10,400 / 1,392 | 0.1248 | 0.1662 | +0.4140 | 0.5267 | 0.0000 | 0.00 |
| **2** | Training Mean Baseline | 2021-10-25..2024-06-06 / 2024-09-09..2025-01-13 | 10,400 / 1,392 | 0.1792 | 0.2170 | 0.0000 | 0.4733 | 0.0000 | 0.00 |
| **2** | Ridge Regression | 2021-10-25..2024-06-06 / 2024-09-09..2025-01-13 | 10,400 / 1,392 | 0.1837 | 0.2246 | -0.0711 | 0.4375 | -0.0934 | -3.60 |
| **2** | Random Forest Regressor | 2021-10-25..2024-06-06 / 2024-09-09..2025-01-13 | 10,400 / 1,392 | 0.1782 | 0.2214 | -0.0403 | 0.4775 | -0.1337 | -5.58 |
| **2** | HistGradientBoosting | 2021-10-25..2024-06-06 / 2024-09-09..2025-01-13 | 10,400 / 1,392 | 0.1765 | 0.2210 | -0.0369 | 0.4789 | -0.0318 | -1.19 |
| **3** | Zero Return Baseline | 2021-10-25..2025-01-13 / 2025-04-17..2025-11-03 | 12,800 / 2,192 | 0.0830 | 0.1092 | +0.0292 | 0.4962 | 0.0000 | 0.00 |
| **3** | Training Mean Baseline | 2021-10-25..2025-01-13 / 2025-04-17..2025-11-03 | 12,800 / 2,192 | 0.0873 | 0.1108 | 0.0000 | 0.5038 | 0.0000 | 0.00 |
| **3** | Ridge Regression | 2021-10-25..2025-01-13 / 2025-04-17..2025-11-03 | 12,800 / 2,192 | 0.0883 | 0.1108 | -0.0002 | 0.5160 | -0.0726 | -3.00 |
| **3** | Random Forest Regressor | 2021-10-25..2025-01-13 / 2025-04-17..2025-11-03 | 12,800 / 2,192 | 0.0879 | 0.1111 | -0.0057 | 0.5117 | +0.0462 | +2.40 |
| **3** | HistGradientBoosting | 2021-10-25..2025-01-13 / 2025-04-17..2025-11-03 | 12,800 / 2,192 | 0.0894 | 0.1123 | -0.0274 | 0.5470 | +0.0772 | +4.07 |

---

## 5. Statistical and Quantitative Analysis

### Why does Logistic Regression outperform complex Boosters in Task A?
Financial asset return time series exhibit an extremely low signal-to-noise ratio ($\approx 1\% - 2\%$). Complex non-linear decision trees (Random Forest, XGBoost, HistGradientBoosting) with high capacity tend to overfit the historical noise patterns in technical indicators, degrading generalization on non-stationary test regimes. L2-regularized Logistic Regression acts as an intrinsic shrinkage prior, extracting the modest linear drift while minimizing variance error.

### Why do all ML regressors produce negative OOS $R^2$ in Task B?
In financial econometrics:
$$\text{OOS } R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y}_{\text{train}})^2}$$
A negative OOS $R^2$ means that the conditional model predictions $\hat{y}_i$ produced higher mean squared error than simply guessing the unconditional historical mean $\bar{y}_{\text{train}}$. 
**Econometric Interpretation & Limitations:** Within this specific universe of 16 NSE equities and across this 4-year sample, the 20 lagged technical features evaluated by these three regression architectures (Ridge, Random Forest, HistGradientBoosting) failed to generate predictive signal superior to naive zero or unconditional mean baselines in terms of out-of-sample ^2$. While consistent with weak-form market efficiency under the joint hypothesis problem, this is specifically an empirical failure of this feature set and model family to extract 3-month predictive alpha, rather than a universal proof of market efficiency. In regime shifts (e.g. Fold 2), market momentum frequently reverses, causing momentum-based regressors to experience severe prediction penalties.

---

## 6. Artifact Inventory and Provenance

All artifacts generated by these real training runs have been saved to the workspace:

| Artifact Type | File Path | Records / Description |
| :--- | :--- | :--- |
| **Development Predictions (Task A)** | `outputs/predictions/development_classification_predictions.csv` | 8,160 rows: out-of-sample predictions & probabilities for all 5 models |
| **Development Predictions (Task B)** | `outputs/predictions/development_regression_predictions.csv` | 4,976 rows: out-of-sample continuous return predictions |
| **Fold-wise Results (Task A)** | `outputs/reports/walk_forward_classification_results.csv` | 15 records (3 folds × 5 models) with full metric vectors |
| **Fold-wise Results (Task B)** | `outputs/reports/walk_forward_regression_results.csv` | 15 records (3 folds × 5 models) with MAE, RMSE, OOS $R^2$, Rank IC |
| **Summary Metrics (Task A)** | `outputs/reports/walk_forward_classification_summary.csv` | Aggregate statistics across folds |
| **Summary Metrics (Task B)** | `outputs/reports/walk_forward_regression_summary.csv` | Aggregate statistics across folds |
| **Experiment Metadata (Task A)** | `outputs/reports/experiment_config_classification.json` | Environment, parameters, date ranges, confusion matrices |
| **Experiment Metadata (Task B)** | `outputs/reports/experiment_config_regression.json` | Environment, parameters, date ranges, baseline comparisons |
| **Fitted Best Model (Task A)** | `outputs/models/classification_best_dev_model.joblib` | Logistic Regression pipeline fitted on full 16,784 dev rows |
| **Fitted Best Model (Task B)** | `outputs/models/regression_best_dev_model.joblib` | Random Forest pipeline fitted on full 16,000 dev rows |

### Environment Details
- **Python:** 3.12.10
- **Scikit-Learn:** 1.8.0
- **XGBoost:** 3.2.0
- **Pandas:** 3.0.1
- **NumPy:** 2.2.3
- **Random Seed:** 42

---

## 7. Recommended Next Steps

1. **Model Selection Decision:**
   - **Task A (Classification):** Select **Logistic Regression** as the primary candidate model based on superior cross-validated ROC-AUC (0.5225) and balanced accuracy (51.41%).
   - **Task B (Regression):** Acknowledge that the naive Zero-Return / Historical Mean baselines outperform ML models in MSE. If an ML regressor is mandated, **Random Forest Regressor** is the closest candidate (MAE 0.1415), while **HistGradientBoosting** offers the only positive Rank IC (+0.0235).

2. **Holdout Evaluation Gate:**
   - The final holdout partition (2026-01-22 to 2026-10-15) remains untainted.
   - Once stakeholders approve the model selection, a single, final evaluation run can be triggered via `--evaluate-holdout`.
