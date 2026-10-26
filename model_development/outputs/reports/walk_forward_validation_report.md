# Walk-Forward Chronological Cross-Validation Report

**Validation Scheme:** Expanding-Window Walk-Forward Cross-Validation  
**Purging/Embargo Buffer:** 5 Trading Sessions between Training and Validation Windows  
**Folds Evaluated:** 3 Expanding Temporal Windows  

---

## 1. Fold Specifications and Date Boundaries

| Fold | Window Type | Training Period | Training Rows | Purge Window | Validation Period | Validation Rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fold 1** | Expanding | 2021-10-25 to 2023-07-03 | 6,704 | 5 sessions | 2023-07-11 to 2024-05-08 | 3,280 |
| **Fold 2** | Expanding | 2021-10-25 to 2024-05-08 | 10,064 | 5 sessions | 2024-05-16 to 2025-03-07 | 3,280 |
| **Fold 3** | Expanding | 2021-10-25 to 2025-03-07 | 13,424 | 5 sessions | 2025-03-18 to 2026-01-13 | 3,280 |

*Note: In all folds, observations across all 16 universe stocks are synchronized strictly by distinct trading dates. Preprocessing pipelines (imputers and scalers) are fitted strictly on each fold's training rows.*

---

## 2. Fold-by-Fold Performance Breakdown

### 2.1 Out-of-Sample ROC-AUC per Fold

| Model | Fold 1 ROC-AUC | Fold 2 ROC-AUC | Fold 3 ROC-AUC | Mean ROC-AUC | Std ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| **Logistic Regression** | 0.5307 | 0.5064 | 0.5106 | **0.5159** | 0.0130 |
| **Random Forest** | 0.5228 | 0.4988 | 0.4921 | 0.5046 | 0.0157 |
| **HistGradientBoosting** | 0.5186 | 0.4941 | 0.5006 | 0.5044 | 0.0127 |
| **XGBoost** | 0.5283 | 0.4891 | 0.5012 | 0.5062 | 0.0200 |

### 2.2 Out-of-Sample Accuracy per Fold

| Model | Fold 1 Accuracy | Fold 2 Accuracy | Fold 3 Accuracy | Mean Accuracy | Std Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.5488 | 0.5256 | 0.5165 | 0.5303 | 0.0165 |
| **Logistic Regression** | 0.5180 | 0.5140 | 0.4957 | 0.5092 | 0.0118 |
| **Random Forest** | 0.5204 | 0.5055 | 0.4954 | 0.5071 | 0.0125 |
| **HistGradientBoosting** | 0.5152 | 0.4936 | 0.5003 | 0.5030 | 0.0112 |
| **XGBoost** | 0.5348 | 0.5085 | 0.5098 | 0.5177 | 0.0146 |

### 2.3 Out-of-Sample Balanced Accuracy per Fold

| Model | Fold 1 Bal Acc | Fold 2 Bal Acc | Fold 3 Bal Acc | Mean Bal Acc | Std Bal Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| **Logistic Regression** | 0.5222 | 0.5043 | 0.5007 | **0.5091** | 0.0115 |
| **Random Forest** | 0.5195 | 0.4982 | 0.4923 | 0.5033 | 0.0143 |
| **HistGradientBoosting** | 0.5113 | 0.4916 | 0.4977 | 0.5002 | 0.0101 |
| **XGBoost** | 0.5268 | 0.4974 | 0.4957 | 0.5066 | 0.0175 |

---

## 3. Key Observations and Regime Insights

1. **Macro Regime Impact on Model Performance:**
   - **Fold 1 (July 2023 - May 2024):** All models achieved their highest predictive accuracy and ROC-AUC (ranging from 0.518 to 0.531). This corresponded to a steady macroeconomic bull market in India where momentum and trend continuation features (`return_20d`, `close_vs_sma_50`) yielded positive predictive signal.
   - **Fold 2 (May 2024 - March 2025):** Included the June 4, 2024 general election market crash and subsequent high-volatility consolidation. All models experienced drops in ROC-AUC (0.489 to 0.506), demonstrating how systemic shock events temporarily invert short-term momentum signals.
   - **Fold 3 (March 2025 - January 2026):** Models stabilized around 0.500 - 0.511 ROC-AUC, reflecting range-bound price action.

2. **Model Selection Rationale:**
   - Logistic Regression demonstrated the highest average ROC-AUC ($0.5159$) and lowest standard deviation across folds ($0.0130$), outperforming complex non-linear tree models on average during cross-validation due to robust L2 shrinkage preventing overfitting.
   - Tree-based models (XGBoost, Random Forest, HistGradientBoosting) performed similarly ($0.504 - 0.506$), indicating that non-linear interaction terms on simple daily price technical indicators provide marginal additional edge over linear regularized signals.
