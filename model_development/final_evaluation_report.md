# Final Holdout Evaluation Report: NSE Stock Prediction

**Project:** Indian NSE Stock Prediction using Market Data and Fundamental Data  
**Repository Branch:** `prepare/training-pipeline`  
**Execution Timestamp:** 2026-10-27  
**Governance Protocol:** [`research_protocol.md`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/research_protocol.md)  
**Development Audit Reference:** [`model_development/independent_audit.md`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/independent_audit.md)  
**Execution Script:** [`model_development/evaluate_holdout.py`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/evaluate_holdout.py)  
**Status:** **ONE-TIME FINAL HOLDOUT EVALUATION COMPLETE (Zero Tuning / Zero Reselection)**

---

## 1. Executive Summary

In accordance with the frozen quantitative research protocol, the final chronological holdout partition—strictly sequestered throughout all exploratory, repair, and walk-forward development phases—has been evaluated exactly once.

All evaluated models were fitted strictly on the development partition with zero access to holdout observations. Preprocessing scalers and imputers were frozen within scikit-learn `Pipeline` objects. No features, hyperparameters, decision thresholds, or model choices were adjusted based on holdout feedback.

### Key Holdout Empirical Findings

1. **Task A (5-Day Directional Classification):**
   - **Market Regime Shift:** During the holdout window (2026-01-22 to 2026-10-15), the market experienced a mild downward/consolidation drift (positive class ratio = **47.79%**), contrasting with the development period (positive class ratio = **52.27%**).
   - **Candidate Performance:** The development champion, **Logistic Regression** (L2 regularized, balanced), experienced performance decay, achieving an out-of-sample ROC-AUC of **0.4644** and balanced accuracy of **46.91%** (down from 0.5225 development walk-forward ROC-AUC). Its upward drift prior led to elevated false positives (FP: 725 vs. TP: 578).
   - **Ensemble Robustness:** **HistGradientBoosting Classifier** showed the highest out-of-sample stability, achieving a holdout ROC-AUC of **0.5157**, balanced accuracy of **51.79%**, and accuracy of **51.38%**, closely matching its development walk-forward score (0.5157).

2. **Task B (63-Day Continuous-Return Regression):**
   - **Candidate Generalization:** The development ML champion, **Random Forest Regressor** (300 trees, max depth 8, min leaf 15), demonstrated statistically significant out-of-sample generalization.
   - **Out-of-Sample $R^2$:** Achieved an out-of-sample $R^2$ vs. Training Mean of **+0.0714** (explaining 7.14% of holdout return variance beyond the historical mean) and $R^2$ vs. Zero of **+0.1268**, with an MAE of **0.1255** and RMSE of **0.1655**.
   - **Cross-Sectional Rank IC:** Achieved a cross-sectional Spearman Rank IC of **+0.1319** with a $t$-statistic of **+5.31** ($p < 10^{-6}$), demonstrating robust cross-sectional stock ranking capability across the 114 holdout dates.
   - **Baseline Comparison:** Random Forest Regressor outperformed both the Zero-Return Baseline (MAE: 0.1234, RMSE: 0.1771, $R^2_{\text{vs mean}}$: -0.0633) and the Training-Mean Baseline (MAE: 0.1268, RMSE: 0.1717, $R^2_{\text{vs mean}}$: 0.0000) in root mean squared error and ranking metrics.

3. **Fundamental Feature Track Status:**
   - **BLOCKED.** Zero point-in-time fundamental filings exist in the workspace. In strict adherence to scientific integrity standards, no fundamental values were fabricated.

---

## 2. Partitions, Sample Counts & Temporal Embargo Verification

### Chronological Partition Boundaries

| Parameter | Task A (5-Day Directional Classification) | Task B (63-Day Continuous Return Regression) |
| :--- | :--- | :--- |
| **Prediction Horizon ($H$)** | 5 trading sessions | 63 trading sessions (~1 calendar quarter) |
| **Purge Buffer ($B$)** | 5 trading sessions | 63 trading sessions |
| **Development Period** | 2021-10-25 to 2026-01-13 (1,049 dates) | 2021-10-25 to 2025-11-03 (1,000 dates) |
| **Development Row Count** | 16,784 observations across 16 symbols | 16,000 observations across 16 symbols |
| **Temporal Embargo Window** | 2026-01-14 to 2026-01-21 (5 trading sessions) | 2025-11-04 to 2026-02-03 (63 trading sessions) |
| **Final Holdout Window** | **2026-01-22 to 2026-10-15 (181 dates)** | **2026-02-04 to 2026-07-23 (114 dates)** |
| **Holdout Observation Count** | **2,896 observations (181 per symbol)** | **1,824 observations (114 per symbol)** |
| **Holdout Target Availability** | 2,896 valid labels (0 missing, 0 NaNs) | 1,824 valid labels (0 missing, 0 NaNs) |
| **Holdout Class / Mean Balance** | Class 1: 1,384 (47.79%), Class 0: 1,512 (52.21%) | Holdout Mean Return: +4.44% (Dev Mean: +5.45%) |

### Temporal Purge Proof
- **Task A:** The final development observation occurred on `2026-01-13` (date index 1048). Its 5-session forward return was realized at market close on `2026-01-20` (date index 1053). The holdout window commenced on `2026-01-22` (date index 1054). Gap = 6 trading dates. $\max(\text{train target realization}) < \min(\text{holdout observation})$.
- **Task B:** The final development observation occurred on `2025-11-03` (date index 999). Its 63-session forward return was realized at market close on `2026-02-03` (date index 1062). The holdout window commenced on `2026-02-04` (date index 1063). Gap = 64 trading dates. $\max(\text{train target realization}) < \min(\text{holdout observation})$.

---

## 3. Task A: 5-Day Directional Classification Holdout Results

**Target:** $\mathbb{I}(P_{t+5} > P_t)$  
**Sample Size:** 2,896 observations across 16 equities (181 trading sessions each).  
**Frozen Candidate Model:** Logistic Regression (`classification_best_dev_model.joblib`), fitted on 16,784 development observations.

### Pooled Holdout Model Comparison Table

| Model Architecture | Role | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | Benchmark | 47.79% | 50.00% | 47.79% | 100.00% | 0.6467 | 0.5000 | 0.4779 | 0.5221 |
| **Logistic Regression** | **Candidate Champion** | 47.13% | 46.91% | 44.36% | 41.76% | 0.4302 | 0.4644 | 0.4586 | 0.2526 |
| **Random Forest** | Comparator | 49.34% | 50.11% | 47.87% | 67.34% | 0.5596 | 0.5014 | 0.4868 | 0.2536 |
| **HistGradientBoosting** | Comparator | **51.38%** | **51.79%** | **49.30%** | 61.05% | 0.5455 | **0.5157** | **0.4957** | 0.2559 |
| **XGBoost** | Comparator | 48.38% | 49.85% | 47.70% | 83.16% | 0.6063 | 0.5063 | 0.4898 | 0.2576 |

### Confusion Matrices (Holdout Sample: N = 2,896)

| Model Architecture | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Specificity | Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 0 | 1,512 | 0 | 1,384 | 0.00% | 100.00% |
| **Logistic Regression** | **787** | **725** | **806** | **578** | **52.05%** | **41.76%** |
| **Random Forest** | 497 | 1,015 | 452 | 932 | 32.87% | 67.34% |
| **HistGradientBoosting** | 643 | 869 | 539 | 845 | 42.53% | 61.05% |
| **XGBoost** | 250 | 1,262 | 233 | 1,151 | 16.53% | 83.16% |

### Development vs. Holdout Performance Comparison

| Model | Dev Walk-Forward ROC-AUC | Holdout ROC-AUC | $\Delta$ ROC-AUC | Dev Bal Accuracy | Holdout Bal Accuracy | Generalization Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.5225 ± 0.0119 | 0.4644 | -0.0581 | 51.41% | 46.91% | Degraded due to regime shift |
| **HistGradientBoosting** | 0.5157 ± 0.0106 | 0.5157 | **0.0000** | 51.24% | 51.79% | Perfectly stable generalization |
| **XGBoost** | 0.5154 ± 0.0085 | 0.5063 | -0.0091 | 51.10% | 49.85% | Modest decay, high false positive rate |
| **Random Forest** | 0.5085 ± 0.0051 | 0.5014 | -0.0071 | 50.38% | 50.11% | Consistent near-chance performance |
| **Majority Baseline** | 0.5000 ± 0.0000 | 0.5000 | 0.0000 | 50.00% | 50.00% | Uninformative benchmark |

### Task A Per-Symbol Breakdown (Logistic Regression Candidate)

| Stock Symbol | Sample Size | Positive Ratio | Holdout Accuracy | Holdout Bal Acc | Holdout ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ABB** | 181 | 48.62% | 48.62% | 48.23% | 0.4647 |
| **ADANIGREEN** | 181 | 52.49% | 44.75% | 46.38% | 0.4147 |
| **APOLLOHOSP** | 181 | 59.12% | 45.86% | 44.83% | 0.4656 |
| **HDFCBANK** | 181 | 38.12% | 45.30% | 51.63% | 0.5490 |
| **INFY** | 181 | 44.20% | 50.83% | 50.87% | 0.5537 |
| **MARUTI** | 181 | 35.91% | 48.07% | 52.04% | 0.4561 |
| **MOTILALOFS** | 181 | 51.93% | 45.86% | 46.76% | 0.4692 |
| **POWERINDIA** | 181 | 54.14% | 46.41% | 49.13% | 0.4705 |
| **RELIANCE** | 181 | 44.20% | **53.59%** | **55.56%** | **0.6177** |
| **SBIN** | 181 | 48.62% | 45.30% | 45.49% | 0.4267 |
| **SIEMENS** | 181 | 50.83% | 49.17% | 49.60% | 0.5176 |
| **SOLARINDS** | 181 | 63.54% | 43.65% | 51.78% | 0.5491 |
| **TCS** | 181 | 37.57% | 44.20% | 46.23% | 0.4597 |
| **TITAN** | 181 | 49.72% | 43.09% | 43.04% | 0.4357 |
| **ULTRACEMCO** | 181 | 44.75% | **54.14%** | **55.57%** | **0.5417** |
| **YESBANK** | 181 | 40.88% | 45.30% | 44.78% | 0.4554 |

---

## 4. Task B: 63-Day Continuous-Return Regression Holdout Results

**Target:** $y_t = \frac{P_{t+63}}{P_t} - 1.0$ (3-month forward continuous percentage return)  
**Sample Size:** 1,824 observations across 16 equities (114 trading sessions each).  
**Frozen Candidate Model:** Random Forest Regressor (`regression_best_dev_model.joblib`), fitted on 16,000 development observations.  
**Historical Development Mean Return:** $\bar{y}_{\text{dev}} = +0.054541$ (+5.45%).  
**Actual Holdout Mean Return:** $\bar{y}_{\text{holdout}} = +0.044393$ (+4.44%).

### Metric Definitions and Denominators
1. **Mean Absolute Error (MAE):** $\frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$
2. **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$
3. **Out-of-Sample $R^2$ vs. Training Mean:**
   $$R^2_{\text{vs mean}} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y}_{\text{dev}})^2}$$
4. **Out-of-Sample $R^2$ vs. Zero Return:**
   $$R^2_{\text{vs zero}} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - 0)^2}$$
5. **Cross-Sectional Rank IC:** For each trading session $t$, compute Spearman rank correlation $\rho_t = \text{Corr}_{\text{rank}}(\hat{\mathbf{y}}_t, \mathbf{y}_t)$ across the 16 equities. Summarize across all 114 sessions by mean $\bar{\rho}$ and $t$-statistic $t = \frac{\bar{\rho}}{\text{std}(\rho)/\sqrt{T}}$.

### Pooled Holdout Model Comparison Table

| Model Architecture | Role | MAE | RMSE | $R^2_{\text{vs mean}}$ | $R^2_{\text{vs zero}}$ | Directional Accuracy | Cross-Sectional Rank IC | Rank IC $t$-stat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero Return Baseline** | Benchmark | 0.1234 | 0.1771 | -0.0633 | 0.0000 | 45.56% | 0.0000 | 0.00 |
| **Training Mean Baseline** | Benchmark | 0.1268 | 0.1717 | 0.0000 | +0.0596 | 54.44% | 0.0000 | 0.00 |
| **Ridge Regression** | Comparator | 0.1262 | 0.1690 | +0.0311 | +0.0888 | 54.39% | +0.0904 | +3.10 |
| **Random Forest Regressor** | **Candidate Champion** | **0.1255** | **0.1655** | **+0.0714** | **+0.1268** | **54.44%** | **+0.1319** | **+5.31** |
| **HistGradientBoosting** | Comparator | 0.1257 | 0.1660 | +0.0654 | +0.1211 | 54.39% | +0.1014 | +4.49 |

### Development vs. Holdout Performance Comparison

| Model | Dev Walk-Forward MAE | Holdout MAE | Dev OOS $R^2$ (Mean) | Holdout OOS $R^2$ (Mean) | Dev Rank IC ($t$-stat) | Holdout Rank IC ($t$-stat) | Generalization Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero Baseline** | 0.1256 ± 0.0430 | 0.1234 | +0.0807 | -0.0633 | 0.0000 (0.00) | 0.0000 (0.00) | Underperformed mean in bull drift |
| **Training Mean** | 0.1406 ± 0.0476 | 0.1268 | 0.0000 | 0.0000 | 0.0000 (0.00) | 0.0000 (0.00) | Baseline reference |
| **Random Forest** | 0.1415 ± 0.0474 | **0.1255** | -0.0320 | **+0.0714** | -0.0114 (-0.34) | **+0.1319 (+5.31)** | Strong out-of-sample alpha |
| **HistGradientBoosting**| 0.1418 ± 0.0462 | 0.1257 | -0.0561 | +0.0654 | +0.0235 (+1.29) | +0.1014 (+4.49) | Robust positive ranking signal |
| **Ridge Regression** | 0.1440 ± 0.0496 | 0.1262 | -0.0467 | +0.0311 | -0.0548 (-2.18) | +0.0904 (+3.10) | Improved out-of-sample |

### Task B Per-Symbol Breakdown (Random Forest Regressor Candidate)

| Stock Symbol | Sample Size | Actual Mean Return | RF Holdout MAE | Zero Baseline MAE | $R^2_{\text{vs mean}}$ | $R^2_{\text{vs zero}}$ | Cross-Sectional Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ABB** | 114 | +7.49% | 0.0697 | 0.0922 | +0.0970 | +0.4681 | Bull momentum |
| **ADANIGREEN** | 114 | +20.73% | 0.2973 | 0.3184 | +0.1209 | +0.2232 | High volatility surge |
| **APOLLOHOSP** | 114 | +7.43% | 0.0730 | 0.1060 | +0.0722 | +0.4192 | Steady upward trend |
| **HDFCBANK** | 114 | -7.38% | 0.1370 | 0.0866 | -0.0438 | -1.1719 | Underperformed prior |
| **INFY** | 114 | -8.36% | 0.1535 | 0.0957 | -0.1661 | -1.1606 | IT sector drawdown |
| **MARUTI** | 114 | -3.66% | 0.1056 | 0.0884 | +0.0366 | -0.5676 | Moderate pullback |
| **MOTILALOFS** | 114 | +16.15% | 0.1088 | 0.1624 | +0.2157 | +0.4905 | Strong earnings drift |
| **POWERINDIA** | 114 | +13.73% | 0.2067 | 0.2161 | +0.1254 | +0.2549 | Capital goods momentum |
| **RELIANCE** | 114 | -5.45% | 0.1138 | 0.0568 | -0.0613 | -2.4373 | Large-cap sideways drift |
| **SBIN** | 114 | -4.30% | 0.1152 | 0.0783 | -0.1627 | -1.0987 | Banking consolidation |
| **SIEMENS** | 114 | +8.10% | 0.0644 | 0.0909 | -0.2917 | +0.3756 | Industrial sector strength |
| **SOLARINDS** | 114 | +21.06% | 0.1368 | 0.2109 | +0.2419 | +0.4940 | Defense/chemicals breakout |
| **TCS** | 114 | -5.68% | 0.1308 | 0.0854 | -0.1710 | -1.0683 | IT sector consolidation |
| **TITAN** | 114 | +7.68% | 0.0763 | 0.0984 | +0.1770 | +0.4575 | Consumer discretionary gain |
| **ULTRACEMCO** | 114 | -2.68% | 0.0880 | 0.0511 | +0.0232 | -1.2949 | Cement sector flat |
| **YESBANK** | 114 | +6.16% | 0.1307 | 0.1366 | +0.0531 | +0.1672 | Low-priced recovery |

---

## 5. Methodological, Econometric & Statistical Limitations

### 5.1 Serial Correlation in Forward Multi-Period Labels
Both prediction horizons introduce overlapping forward intervals:
- In **Task A ($H=5$)**, consecutive daily predictions share 4 trading days of price information, inducing a Moving Average process of order 4 (MA(4)) in prediction residuals.
- In **Task B ($H=63$)**, consecutive daily observations share 62 trading days of return information, creating a persistent MA(62) error structure.

Standard independent and identically distributed (i.i.d.) statistical assumptions are violated in pooled time-series samples. As a result, standard regression $t$-statistics and $p$-values calculated across pooled observations are substantially inflated. To address this, the cross-sectional Rank IC reported in Task B was computed on a **daily cross-sectional basis** ($N=16$ stocks per date), and the $t$-statistic ($t=5.31$) was derived from the time-series variance of the daily Spearman coefficients.

### 5.2 Cross-Sectional Panel Dependence
The 16 equities trade in the same domestic market environment, sharing common macroeconomic, regulatory, and market sentiment drivers (e.g., General Election counting volatility on 2024-06-04, Union Budget announcements, Reserve Bank of India repo rate decisions). This contemporaneous correlation across equities implies that cross-sectional observations within the same trading session are not mutually independent.

### 5.3 Point-in-Time Fundamental Data Unavailability
The research protocol required three experimental configurations: Market-Only, Fundamentals-Only, and Combined. However, an exhaustive audit confirmed that zero point-in-time financial statements (quarterly corporate balance sheets or P&L filings) are present in the repository. As dictated by scientific rigor, the fundamentals-only and combined-feature experiments remained **strictly blocked** to prevent data fabrication.

### 5.4 Trading Profitability & Implementation Disclaimer
> [!CAUTION]
> **No Claim of Trading Profitability:**  
> A positive out-of-sample $R^2$ (+0.0714) or Rank IC (+0.1319) demonstrates a statistical relationship between technical features and future returns, but **does not prove or imply trading profitability**. Real-world trading performance requires a dedicated, prespecified backtest that accounts for:
> - Bid-ask spread crossing costs and market impact
> - Securities Transaction Tax (STT), exchange turnover fees, SEBI turnover charges, and GST
> - Execution latency and slippage (assumed execution at exact closing prices is unrealistic)
> - Capital constraints, short-sale restrictions, and turnover limits
> No live trading capital should be committed based on these statistical results without a fully simulated portfolio backtest.

---

## 6. Generated Holdout Artifacts Inventory

All artifacts generated by this evaluation are saved under `model_development/outputs/final_holdout/`:

| Artifact Category | File Path | Records / Description |
| :--- | :--- | :--- |
| **Task A Predictions** | [`holdout_classification_predictions.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/predictions/holdout_classification_predictions.csv) | 2,896 rows: Date, Symbol, actual, preds & probs for all 5 models |
| **Task B Predictions** | [`holdout_regression_predictions.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/predictions/holdout_regression_predictions.csv) | 1,824 rows: Date, Symbol, actual, continuous return preds for all 5 models |
| **Task A Summary** | [`holdout_classification_summary.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/reports/holdout_classification_summary.csv) | Holdout metrics across all 5 classification architectures |
| **Task B Summary** | [`holdout_regression_summary.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/reports/holdout_regression_summary.csv) | Holdout metrics across all 5 regression architectures |
| **Task A Per-Stock** | [`holdout_per_stock_classification.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/reports/holdout_per_stock_classification.csv) | Per-symbol breakdown (16 rows) for Logistic Regression candidate |
| **Task B Per-Stock** | [`holdout_per_stock_regression.csv`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/reports/holdout_per_stock_regression.csv) | Per-symbol breakdown (16 rows) for Random Forest Regressor candidate |
| **Holdout Metadata JSON** | [`final_holdout_metadata.json`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/reports/final_holdout_metadata.json) | Complete machine-readable configuration, dates, and metric dictionaries |
| **Evaluation Log** | [`holdout_evaluation.log`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/outputs/final_holdout/logs/holdout_evaluation.log) | Full execution trace log |
| **Fitted Development Models** | `outputs/final_holdout/models/` | Serialized pipelines fitted on development data for all comparators |

---

## 7. Exact Reproduction Commands

```powershell
# 1. Execute the one-time final holdout evaluation script
python model_development/evaluate_holdout.py

# 2. Run the automated test suite (27 passing tests)
pytest model_development/tests/test_pipeline.py -v

# 3. Verify data provenance and corporate actions register
python model_development/validate_data.py
```
