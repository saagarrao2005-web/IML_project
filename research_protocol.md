# Frozen Quantitative Research Protocol

**Project Title:** Indian NSE Stock Prediction using Market Data and Fundamental Data  
**Repository:** `https://github.com/saagarrao2005-web/IML_project`  
**Protocol Version:** 1.0 (Frozen Specification)  
**Effective Date:** 27 October 2026  
**Auditor:** Senior Quantitative Researcher & ML Validation Specialist  

---

## 1. Research Objectives & Universe Scope

This quantitative research protocol defines the frozen experimental parameters, data standards, feature construction rules, temporal validation partitions, and evaluation benchmarks for predicting stock returns across major National Stock Exchange of India (NSE) equities.

### 1.1 Target Equities Universe
The evaluation universe consists of **16 major NSE equities**:
`ABB`, `ADANIGREEN`, `APOLLOHOSP`, `HDFCBANK`, `INFY`, `MARUTI`, `MOTILALOFS`, `POWERINDIA`, `RELIANCE`, `SBIN`, `SIEMENS`, `SOLARINDS`, `TCS`, `TITAN`, `ULTRACEMCO`, `YESBANK`.

- **Series Filter:** Strictly `EQ` (ordinary equity shares). All non-EQ series (`BE`, `BL`, `BZ`, `IL`, `N1`-`N6`, `SM`) are excluded.
- **Sample Timeframe:** 25 October 2021 to 23 October 2026 (5 full calendar years, 1,240 uniform trading sessions per equity, 19,840 total daily observations).

---

## 2. Formal Prediction Tasks

The research protocol separates the quantitative prediction problem into **two independent tasks**. Under no circumstances may their targets, evaluation samples, frequencies, or loss functions be conflated.

### Task A: Short-Term Directional Classification (Primary Operational Task)
- **Prediction Objective:** Predict whether the corporate-action-adjusted closing price 5 trading sessions ahead ($t+5$) is strictly higher than the adjusted closing price at session $t$.
- **Sampling Frequency:** Daily (every valid exchange trading session).
- **Prediction Timestamp:** At market close of session $t$ ($15:30$ IST).
- **Feature Cutoff:** Strictly market and financial data finalized and publicly available at or before $15:30$ IST on date $t$.
- **Mathematical Target Definition:**
  $$R_{t, t+5} = \frac{P_{t+5}^{\text{adj}}}{P_t^{\text{adj}}} - 1$$
  $$y_{t}^{(5)} = \begin{cases} 1.0 & \text{if } R_{t, t+5} > 0 \\ 0.0 & \text{if } R_{t, t+5} \le 0 \end{cases}$$
  *(Ties where $P_{t+5}^{\text{adj}} = P_t^{\text{adj}}$ are assigned to the 0.0 class).*
- **Missing Forward Outcomes:** The final 5 trading sessions for each equity have no forward exit price within the dataset and are strictly labeled `NaN` and sequestered from training and evaluation.

### Task B: Medium-Term Continuous-Return Regression (Research Horizon Task)
- **Prediction Objective:** Predict the continuous corporate-action-adjusted percentage return over a 3-month forward horizon ($H = 63$ trading sessions, approximately 1 calendar quarter).
- **Sampling Frequency:** Monthly evaluation frequency (21-session sampling or rolling monthly steps).
- **Prediction Timestamp:** At market close of evaluation session $t$ ($15:30$ IST).
- **Mathematical Target Definition:**
  $$y_{t}^{(63)} = R_{t, t+63} = \frac{P_{t+63}^{\text{adj}}}{P_t^{\text{adj}}} - 1$$
- **Missing Forward Outcomes:** The final 63 trading sessions for each equity have no forward exit price within the dataset and are strictly labeled `NaN` and excluded from model fitting.

---

## 3. Feature Families & Ablation Protocols

Models must be trained and compared across three strictly defined feature configurations:

### Configuration 1: Market-Only Features (20 Predictors)
- **Momentum Multi-Horizon Returns:** `return_1d`, `return_2d`, `return_5d`, `return_10d`, `return_20d`
- **Trend Distance Ratios:** `close_vs_sma_5`, `close_vs_sma_10`, `close_vs_sma_20`, `close_vs_sma_50`
- **Return Volatility:** `volatility_10d`, `volatility_20d` (rolling standard deviation of daily return)
- **Intraday Price Structure:** `daily_range_pct` ($(H_t - L_t)/P_t$), `open_close_return` ($P_t / O_t - 1$)
- **Volume & Liquidity:** `volume_change_1d`, `volume_vs_avg20`, `trades_vs_avg20`
- **Delivery Quality:** `delivery_pct` ($\text{Deliverable Qty} / \text{Traded Qty} \times 100$)
- **Technical Oscillator:** `rsi_14` (Wilder's 14-period Relative Strength Index, warmup NaNs preserved for imputer)
- **Calendar Identifiers:** `day_of_week` (0 to 4), `month` (1 to 12)

### Configuration 2: Fundamentals-Only Features
- **Standard Fundamental Factors:** P/E Ratio, P/B Ratio, Debt-to-Equity, Return on Equity (ROE), Operating Margin, Quarterly EPS YoY Growth.
- **Point-in-Time Availability Requirement:** A fundamental metric from fiscal period $Q$ can only be attached to observation date $t$ if the corporate financial results publication date $T_{\text{pub}} \le t$. Attaching statements based on quarter-end dates (e.g. using March 31 numbers on April 1) constitutes fatal look-ahead bias.
- **Coverage Prerequisite:** Raw XBRL or regulatory exchange filings must be present. If unavailable, values must **never** be fabricated or silently imputed from future statements.

### Configuration 3: Combined Features (Market + Fundamental)
- Merged feature matrix combining Configurations 1 and 2 evaluated on **identically matched observation subsets** where both market and verified fundamental data are available.

---

## 4. Chronological Validation & Purging Protocol

To prevent look-ahead bias and forward-label leakage, random splits and standard k-fold cross-validation are strictly prohibited.

1. **Expanding-Window Walk-Forward Cross-Validation (3 Folds):**
   - Executed strictly within the Development Set (first 85% of distinct trading dates: 2021-10-25 to 2026-01-13).
   - Fold 1: Train 50% $\to$ Purge $H$ sessions $\to$ Validate 50% to 65%
   - Fold 2: Train 65% $\to$ Purge $H$ sessions $\to$ Validate 65% to 80%
   - Fold 3: Train 80% $\to$ Purge $H$ sessions $\to$ Validate 80% to 100%
2. **Purging & Embargo Mandate:**
   - For Task A ($H=5$): An embargo of 5 trading sessions is enforced between training and validation/holdout sets.
   - For Task B ($H=63$): An embargo of 63 trading sessions is enforced between training and validation/holdout sets.
3. **Sequestered Holdout Partition:**
   - The final 15% of trading dates (2026-01-22 to 2026-10-15, 2,896 observations across 16 equities) is strictly sequestered until models and pipeline hyperparameters are frozen.
4. **Isolated Preprocessing:**
   - Missing value imputation (`SimpleImputer(strategy="median")`) and feature standardization (`StandardScaler()`) must be encapsulated in scikit-learn `Pipeline` objects and fitted **strictly on training fold observations**.

---

## 5. Required Model Architectures & Baselines

### For Continuous-Return Regression (Task B & Secondary Regression):
1. **Zero-Return Naive Baseline:** Predicts $\hat{y}_t = 0.0$ for all observations.
2. **Training-Mean Naive Baseline:** Predicts $\hat{y}_t = \bar{y}_{\text{train}}$ (the mean return of the training split).
3. **Ridge Regression:** L2-regularized linear model ($\alpha = 1.0$).
4. **Random Forest Regressor:** Non-linear bagged ensemble (300 trees, max depth 8, min samples leaf 15).
5. **HistGradientBoosting Regressor:** Histogram-based gradient boosting (learning rate 0.03, max depth 4, min samples leaf 20, L2 reg 2.0).

### For Directional Classification (Task A):
1. **Majority-Class Dummy Baseline:** Predicts the most frequent class in the training split.
2. **Logistic Regression:** L2-regularized linear classifier ($C = 0.1$, balanced class weights).
3. **Random Forest Classifier:** Bagged tree ensemble (300 trees, max depth 8, min samples leaf 15, balanced class weights).
4. **HistGradientBoosting Classifier:** Boosted tree classifier (150 trees, learning rate 0.04, max depth 5, balanced class weights).

---

## 6. Official Evaluation Metrics

All metrics must be computed from raw out-of-sample predictions and never hardcoded:

### Classification Metrics:
- **ROC-AUC & PR-AUC:** Discrimination and precision-recall ranking capability.
- **Balanced Accuracy:** Average recall of Class 1 and Class 0: $\frac{1}{2}(\text{Recall}_1 + \text{Recall}_0)$.
- **Accuracy, Precision, Recall, F1 Score:** At standard probability threshold $p \ge 0.50$.
- **Brier Score:** Mean squared probability error $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$.

### Regression Metrics:
- **MAE:** Mean Absolute Error $\frac{1}{N}\sum |\hat{y}_i - y_i|$.
- **RMSE:** Root Mean Squared Error $\sqrt{\frac{1}{N}\sum (\hat{y}_i - y_i)^2}$.
- **Out-of-Sample $R^2$ (Benchmark: Training Mean):**
  $$R_{\text{OOS}}^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y}_{\text{train}})^2}$$
- **Out-of-Sample $R^2$ (Benchmark: Zero Return):**
  $$R_{\text{zero}}^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - 0)^2}$$
- **Directional Accuracy:** Fraction of instances where $\text{sign}(\hat{y}_i) == \text{sign}(y_i)$.
- **Cross-Sectional Rank IC (Spearman correlation):** Calculated cross-sectionally across stocks on each evaluation date and summarized by mean and t-statistic across time.
