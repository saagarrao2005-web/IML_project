# Quantitative Machine Learning Training Pipeline Design

**Project Title:** Indian NSE Stock Prediction using Market Data and Fundamental Data  
**Repository:** `https://github.com/saagarrao2005-web/IML_project`  
**Document Date:** 27 October 2026  
**Architect:** Senior ML Systems Engineer & Quantitative Researcher  

---

## 1. System Architecture Overview

The training pipeline is structured around strict quantitative finance principles: zero look-ahead bias, rigorous temporal partitioning, isolated preprocessing, and distinct task isolation.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        RAW NSE BHAVCOPY DATA                           │
│                 16 Equities · 1,240 Sessions Each                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CORPORATE ACTION BACK-ADJUSTMENT                     │
│         4 Mechanical Events Adjusted (Bonus / Demerger / Splits)       │
│                  Price & Volume Invariance Verified                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  LEAK-FREE FEATURE & TARGET EXTRACTION                 │
│         20 Technical Features strictly from t <= 15:30 IST             │
│   Task A: 5-Session Target (y_5d)   │  Task B: 63-Session Target (y_63d)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
│  DEVELOPMENT SET (First 85% Dates)  │ │   UNTOUCHED HOLDOUT (Final 15%)     │
│   2021-10-25 to 2026-01-13 (1,049 d) │ │   2026-01-22 to 2026-10-15 (181 d)  │
│         16,784 Observations         │ │         2,896 Observations          │
└──────────────────┬──────────────────┘ └──────────────────┬──────────────────┘
                   │                                       │
                   │ ◄─── Purge Embargo Buffer (H days) ───┤
                   ▼                                       │
┌─────────────────────────────────────┐                    │
│   WALK-FORWARD EXPANDING WINDOW CV  │                    │
│    Fold 1: Train 50% -> Val 15%     │                    │
│    Fold 2: Train 65% -> Val 15%     │                    │
│    Fold 3: Train 80% -> Val 20%     │                    │
│   (Purge H sessions in every fold)  │                    │
└──────────────────┬──────────────────┘                    │
                   │                                       │
                   ▼                                       │
┌─────────────────────────────────────┐                    │
│       MODEL SELECTION DECISION      │                    │
│   Highest Mean Walk-Forward Metric  │                    │
└──────────────────┬──────────────────┘                    │
                   │                                       │
                   ▼                                       │
┌──────────────────────────────────────────────────────────┴──────────────────┐
│                   FINAL RE-FIT & UNTOUCHED HOLDOUT EVALUATION               │
│               Fit best pipeline on full 85% Development Set                 │
│                 Evaluate on Sequestered 15% Holdout Window                  │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION ARTIFACTS GENERATION                       │
│      Model Comparison Table · Calibration Curves · ROC/PR Plots             │
│      Per-Stock Performance Breakdown · Machine-Readable JSON Reports        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Preprocessing & Leakage Elimination

### 2.1 The Container Pipeline Rule
All preprocessing is encapsulated strictly inside scikit-learn `Pipeline` objects:
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", candidate_estimator),
])
```
- **Fit Isolation:** The pipeline's `fit()` method is invoked exclusively on training fold subsets. The validation and holdout sets are transformed via `transform()` using parameters learned strictly from past data.
- **Median Imputation:** Missing feature values (primarily early warmup sessions for 50-day rolling averages) are imputed using training fold medians. At no point can validation or test statistics shift the imputed values.

### 2.2 Purging & Embargo Design
Consecutive daily observations with an $H$-session forward-looking target share $H-1$ sessions of overlapping price information. Evaluating a model on session $t+1$ after training on session $t$ would leak the training target directly into the evaluation set.

- **Task A (5-day horizon):** A strict **5-trading-session purge window** is enforced between training fold $t_{\max}$ and validation $t_{\min}$, as well as between development set and holdout set.
  $$\Delta t_{\text{purge}} = t_{\min, \text{val}} - t_{\max, \text{train}} \ge 5 \text{ trading sessions}$$
- **Task B (63-day horizon):** A strict **63-trading-session purge window** (~3 calendar months) is enforced.

---

## 3. Model Suite Specifications

### 3.1 Task A: Directional Classification Models

| Model Identifier | Architecture | Hyperparameters | Purpose |
| :--- | :--- | :--- | :--- |
| `Majority_Class_Baseline` | DummyClassifier | `strategy="most_frequent"` | Zero-skill classification floor |
| `Logistic_Regression` | Penalized GLM | `C=0.1, penalty="l2", class_weight="balanced"` | Interpretable linear benchmark |
| `Random_Forest` | Bagged Ensemble | `n_estimators=300, max_depth=8, min_samples_leaf=15, class_weight="balanced"` | Non-linear tree ensemble |
| `Hist_Gradient_Boosting` | Histogram Boosted Trees | `max_iter=150, max_depth=5, min_samples_leaf=20, l2_reg=1.5, lr=0.04, class_weight="balanced"` | Fast adaptive non-linear model |
| `XGBoost` | Regularized GBDT | `n_estimators=150, max_depth=4, lr=0.03, subsample=0.8, colsample=0.8, reg_alpha=0.1, reg_lambda=1.0` | High-capacity gradient boosting |

### 3.2 Task B: Continuous-Return Regression Models

| Model Identifier | Architecture | Hyperparameters | Purpose |
| :--- | :--- | :--- | :--- |
| `Zero_Return_Baseline` | Constant Predictor | $\hat{y}_t = 0.0$ | No-change market benchmark |
| `Training_Mean_Baseline` | Constant Predictor | $\hat{y}_t = \bar{y}_{\text{train}}$ | Unconditional expectation floor |
| `Ridge_Regression` | Regularized Linear | $\alpha = 1.0$, standardized features | Linear continuous return model |
| `Random_Forest_Regressor` | Bagged Tree Ensemble | `n_estimators=300, max_depth=8, min_samples_leaf=15` | Non-linear interaction model |
| `Hist_Gradient_Boosting_Regressor` | Histogram GBDT Regressor | `max_iter=150, max_depth=4, min_samples_leaf=20, l2_reg=2.0, lr=0.03` | Non-linear regression baseline |

---

## 4. Evaluation Metrics & Statistical Baselines

### 4.1 Directional Classification Metrics
- **ROC-AUC & PR-AUC:** Rank-order discrimination power.
- **Balanced Accuracy:** $\frac{1}{2}(\text{Recall}_{\text{Up}} + \text{Recall}_{\text{Down}})$. Accounts for minor class imbalance without threshold manipulation.
- **Accuracy, Precision, Recall, F1:** At decision threshold $p \ge 0.50$.
- **Brier Score:** Mean squared error of predicted probabilities: $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$. A random 50/50 model achieves $0.25$.

### 4.2 Continuous-Return Regression Metrics
- **MAE & RMSE:** Mean Absolute Error and Root Mean Squared Error.
- **Out-of-Sample $R^2$ (Benchmark: Training Mean):**
  $$R_{\text{OOS, mean}}^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y}_{\text{train}})^2}$$
  *Positive $R^2$ indicates true predictive skill superior to the unconditional training set mean.*
- **Out-of-Sample $R^2$ (Benchmark: Zero Return):**
  $$R_{\text{OOS, zero}}^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - 0)^2}$$
- **Directional Accuracy:** $\frac{1}{N}\sum \mathbb{I}(\text{sign}(\hat{y}_i) == \text{sign}(y_i))$.
- **Cross-Sectional Rank IC (Spearman Correlation):**
  For each evaluation date $t$, calculate the Spearman rank correlation across all $N$ equities:
  $$\text{Rank IC}_t = \text{Corr}_{\text{rank}}(\hat{y}_{t, \cdot}, y_{t, \cdot})$$
  Summarize by mean $\overline{\text{Rank IC}}$ and t-statistic:
  $$t_{\text{IC}} = \frac{\overline{\text{Rank IC}}}{\text{Std}(\text{Rank IC}) / \sqrt{T}}$$

---

## 5. Feature Ablation Framework

The pipeline provides native feature subset resolution:
1. **Market-Only (`market`):** The 20 verified technical indicators.
2. **Fundamentals-Only (`fundamentals`):** Financial ratios (P/E, P/B, EPS growth, margin, debt-to-equity, ROE).
   - If raw filings are uncommitted in the workspace, the pipeline returns `BLOCKED` and halts execution rather than inventing values.
3. **Combined (`combined`):** Merged market and fundamental features evaluated on matched dates.
   - If fundamental features are unavailable, the pipeline records `BLOCKED` status and prevents unverified runs.
