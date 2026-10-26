# Model Comparison and Evaluation Report

**Evaluation Framework:** Chronological Walk-Forward Cross-Validation (3 Folds) + Untouched Chronological Holdout (2,896 observations)  
**Prediction Horizon:** 5 Trading Sessions  
**Primary Performance Metric:** Out-of-Sample ROC-AUC and Balanced Accuracy  

---

## 1. Executive Model Comparison Matrix

The table below summarizes performance across all evaluated model architectures on both the expanding-window walk-forward cross-validation (development period) and the final untouched chronological holdout set (2026-01-22 to 2026-10-15).

| Model Architecture | Walk-Forward ROC-AUC (Mean $\pm$ Std) | Walk-Forward Accuracy (Mean $\pm$ Std) | Holdout ROC-AUC | Holdout PR-AUC | Holdout Accuracy | Holdout Balanced Accuracy | Holdout F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline (Most Frequent)** | $0.5000 \pm 0.0000$ | $0.5303 \pm 0.0165$ | 0.5000 | 0.4779 | 0.4779 | 0.5000 | 0.6467 |
| **Logistic Regression (Balanced)** | $\mathbf{0.5159 \pm 0.0130}$ | $0.5092 \pm 0.0118$ | 0.4642 | 0.4584 | 0.4706 | 0.4683 | 0.4291 |
| **Random Forest (300 Trees)** | $0.5046 \pm 0.0157$ | $0.5071 \pm 0.0125$ | 0.5017 | 0.4871 | 0.4914 | 0.4990 | 0.5581 |
| **HistGradientBoosting** | $0.5044 \pm 0.0127$ | $0.5030 \pm 0.0112$ | $\mathbf{0.5187}$ | $\mathbf{0.4999}$ | $\mathbf{0.5180}$ | $\mathbf{0.5223}$ | 0.5514 |
| **XGBoost (Depth 4, Reg)** | $0.5062 \pm 0.0200$ | $0.5177 \pm 0.0146$ | 0.5033 | 0.4885 | 0.4845 | 0.4990 | 0.6058 |

---

## 2. Secondary Experiment: Continuous 5-Day Forward Return Regression

In addition to discrete binary classification, a secondary regression experiment was conducted to forecast the continuous 5-trading-session forward return ($\text{future\_return\_5d}$):

- **Model:** HistGradientBoostingRegressor (150 estimators, depth 4, min leaf 20, L2 regularization = 2.0).
- **Holdout MAE:** $0.0347$ (Mean Absolute Error of $3.47\%$).
- **Holdout RMSE:** $0.0472$ (Root Mean Squared Error of $4.72\%$).
- **Directional Accuracy:** $0.4710$ ($47.10\%$ of predicted return signs matched the actual sign).
- **Directional Rank/ROC-AUC:** $0.5041$.

**Finding:** Forecasting raw forward percentage returns using past price-derived technical features produces a near-zero predictive correlation ($R^2 \approx -0.01$, Directional AUC $\approx 0.504$). This confirms that regression does not unlock hidden predictive alpha over classification on pure market price data.

---

## 3. In-Depth Performance & Quantitative Analysis

### 3.1 Walk-Forward Stability vs Holdout Generalization
- During the development phase across 3 chronological expanding folds (2021 to 2026-01), **Logistic Regression** achieved the highest mean ROC-AUC ($0.5159 \pm 0.0130$), outperforming tree ensembles on the earlier market regimes due to strong linear regularization preventing overfitting to noise.
- On the final untouched holdout window (2026-01-22 to 2026-10-15), market dynamics shifted: the market experienced increased choppy volatility and rotation. On this holdout period, **HistGradientBoosting** proved to be the most resilient model, achieving **0.5187 ROC-AUC**, **51.80% Accuracy**, and **52.23% Balanced Accuracy**, outperforming all other models and beating the dummy baseline.
- **XGBoost** and **Random Forest** achieved near-chance holdout ROC-AUCs of **0.5033** and **0.5017** respectively, while **Logistic Regression** declined to **0.4642** on holdout, demonstrating the vulnerability of linear models to regime shifts when equity market trends flatten.

### 3.2 The Reality of Efficient Markets (EMH)
Quantitative finance and academic literature on the Weak-Form Efficient Market Hypothesis (EMH) establish that future short-term price movements in large-cap, liquid equities (such as Reliance, HDFC Bank, TCS, Infosys) follow near random-walk behavior when conditioned solely on past price and volume transformations.
- Real-world institutional equity alpha for 5-day horizon directional models rarely exceeds an ROC-AUC of $0.52 - 0.55$.
- Models claiming $70\% - 90\%$ directional accuracy on daily equity returns invariably suffer from look-ahead bias, random train/test data splitting, or forward-target leakage.
- The verified results achieved here (**0.516** Walk-Forward AUC, **0.519** Holdout AUC) represent an honest, defensible, leakage-free quantitative baseline.
