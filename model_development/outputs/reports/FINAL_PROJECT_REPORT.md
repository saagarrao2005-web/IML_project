# Final Comprehensive Project Report: Directional Prediction of NSE Equities using Machine Learning

**Academic Project:** B.Tech Semester Project — Introduction to Machine Learning (IML)  
**Project Title:** Directional Movement Prediction of National Stock Exchange (NSE) Equities Using Chronological Machine Learning Pipelines  
**Author / Engineering Team:** Quantitative Research & Machine Learning Engineering Team  
**Evaluation Date:** 26 October 2026  
**Target Universe:** 16 Major National Stock Exchange (NSE) Equities  
**Sample Period:** 25 October 2021 to 23 October 2026 (5 Full Calendar Years; 1,240 Trading Sessions per Stock; 19,840 Total Observations)  
**Analytical Dataset Checksum (SHA-256):** `ab9432a93a37f417171c7a105827c5f63568f30dc5af4601a6613a38cf653ccc`  
**Saved Model Artifact Checksum (SHA-256):** `0879c534fca60850e283d77632ad82e911b0ca20e9c300c19918e905a2bdadb3`  

---

## 1. Abstract

This research project develops, audits, and evaluates an end-to-end, leakage-free quantitative machine learning system for predicting the 5-trading-session directional price movement ($t$ to $t+5$) of 16 highly traded equities listed on the National Stock Exchange of India (NSE). Spanning 19,840 daily market observations across five calendar years (October 2021 to October 2026), the study implements a robust quantitative data pipeline addressing multi-series exchange file contamination, mechanical corporate action discontinuities (bonus issues and demergers), and look-ahead information leakage. 

Using an expanding-window chronological walk-forward cross-validation protocol (3 temporal folds) with a strict 5-trading-session purge/embargo buffer, five model architectures were benchmarked: a Majority-Class Dummy Baseline, L2-Regularized Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost. An untouched holdout set of 2,896 observations (22 January 2026 to 15 October 2026) was strictly sequestered until model freezing. On development walk-forward cross-validation, Logistic Regression achieved the highest mean out-of-sample ROC-AUC of $0.5159 \pm 0.0130$. On the untouched holdout period, HistGradientBoosting achieved the highest generalization score with an ROC-AUC of $0.5187$ and Accuracy of $51.80\%$. A Date-Level Moving-Block Bootstrap ($L=5$ dates, 2,000 replications) demonstrated that holdout performance spans 0.50 (95% CI: $[0.4783, 0.5586]$, $p = 0.1790$), establishing that short-term technical predictability in liquid large-cap NSE equities is statistically indistinguishable from random-walk behavior. The project establishes an academically defensible, methodologically rigorous quantitative baseline for financial machine learning research.

---

## 2. Problem Statement and Objectives

### 2.1 Problem Formulation
Predicting asset price trajectories is one of the most challenging problems in applied empirical machine learning due to non-stationarity, regime shifts, and extremely low signal-to-noise ratios. Academic literature and commercial trading claims frequently report directional accuracies of 70% to 90%. However, virtually all such findings are artifacts of catastrophic methodological flaws:
1. Random train-test splitting of time-series data (which leaks future information backwards).
2. Fitting global imputers, scalers, or feature selectors across entire datasets.
3. Failing to apply an embargo/purge buffer around overlapping forward-target return windows.
4. Failing to adjust for mechanical corporate actions (stock splits, bonus issues, and spin-offs).

### 2.2 Core Project Objectives
1. **Data Integrity:** Ingest and normalize 16 NSE equity daily Bhavcopy datasets, eliminate multi-series duplication, and enforce strict OHLC price and volume boundaries.
2. **Corporate Action Reconciliation:** Identify, audit via primary regulatory sources, and retrospectively adjust historical prices and volumes for all stock splits, bonus issues, and demergers without leaking future information into historical features.
3. **Leakage-Safe Feature Engineering:** Engineer 20 predictive features strictly using information known on or before market close on session $t$.
4. **Chronological Validation:** Implement expanding-window walk-forward cross-validation with exact purge buffers and an untouched chronological holdout set.
5. **Rigorous Benchmarking & Statistical Testing:** Benchmark diverse linear and non-linear models against a majority baseline, and conduct block bootstrap significance tests accounting for cross-sectional and temporal serial dependence.
6. **Production Deployment Interface:** Deliver a CLI and programmatic inference module that validates input data freshness and generates directional forecasts with transparent probabilistic uncertainty.

---

## 3. Dataset Description and Limitations

### 3.1 Universe Specification
The target universe comprises 16 liquid large-cap equities representing major sectors of the Indian economy:

| Symbol | Company Name | Industry Sector | Date Range | Observations |
| :--- | :--- | :--- | :--- | :--- |
| **ABB** | ABB India Ltd | Heavy Electrical Equipment / Capital Goods | 2021-10-25 to 2026-10-23 | 1,240 |
| **ADANIGREEN** | Adani Green Energy Ltd | Renewable Power / Utilities | 2021-10-25 to 2026-10-23 | 1,240 |
| **APOLLOHOSP** | Apollo Hospitals Enterprise Ltd | Healthcare Facilities & Services | 2021-10-25 to 2026-10-23 | 1,240 |
| **HDFCBANK** | HDFC Bank Ltd | Banking & Financial Services | 2021-10-25 to 2026-10-23 | 1,240 |
| **INFY** | Infosys Ltd | IT Consulting & Software Services | 2021-10-25 to 2026-10-23 | 1,240 |
| **MARUTI** | Maruti Suzuki India Ltd | Passenger Automobiles | 2021-10-25 to 2026-10-23 | 1,240 |
| **MOTILALOFS** | Motilal Oswal Financial Services | Investment Banking & Retail Broking | 2021-10-25 to 2026-10-23 | 1,240 |
| **POWERINDIA** | Hitachi Energy India Ltd | Power Transmission & Grid Automation | 2021-10-25 to 2026-10-23 | 1,240 |
| **RELIANCE** | Reliance Industries Ltd | Diversified Energy, Retail & Telecom | 2021-10-25 to 2026-10-23 | 1,240 |
| **SBIN** | State Bank of India | Public Sector Banking & Lending | 2021-10-25 to 2026-10-23 | 1,240 |
| **SIEMENS** | Siemens Ltd | Industrial Automation & Drive Tech | 2021-10-25 to 2026-10-23 | 1,240 |
| **SOLARINDS** | Solar Industries India Ltd | Industrial Explosives & Defence | 2021-10-25 to 2026-10-23 | 1,240 |
| **TCS** | Tata Consultancy Services Ltd | Enterprise IT Services & Consulting | 2021-10-25 to 2026-10-23 | 1,240 |
| **TITAN** | Titan Company Ltd | Consumer Discretionary & Jewellery | 2021-10-25 to 2026-10-23 | 1,240 |
| **ULTRACEMCO** | UltraTech Cement Ltd | Building Materials & Grey Cement | 2021-10-25 to 2026-10-23 | 1,240 |
| **YESBANK** | YES Bank Ltd | Retail & Corporate Commercial Banking | 2021-10-25 to 2026-10-23 | 1,240 |

### 3.2 Dataset Schema and Available Columns
Each original CSV provides daily Bhavcopy market summary fields:
- `Symbol`, `Series`, `Date`
- `Prev Close`, `Open Price`, `High Price`, `Low Price`, `Last Price`, `Close Price`, `Average Price`
- `Total Traded Quantity` (Volume), `Turnover ₹`, `No. of Trades`
- `Deliverable Qty`, `% Dly Qt to Traded Qty` (Delivery Percentage)

### 3.3 Intrinsic Dataset Limitations
1. **Market Data Only:** The dataset does not contain quarterly corporate financial statements, earnings surprises, balance sheet metrics, or valuation ratios.
2. **Absence of Alternative / Order Flow Data:** It lacks intraday limit order book data, bid-ask spreads, institutional investor net flows (FII/DII), and news sentiment.
3. **Survivorship Bias:** The 16 equities represent surviving large-cap constituents as of 2026. While chronological out-of-sample partitioning prevents in-sample look-ahead leakage, the universe does not account for companies delisted during 2021–2026.

---

## 4. Data Cleaning and Corporate-Action Treatment

### 4.1 Resolution of Multi-Series Duplication
The raw exchange files contained multiple series codes (`EQ`, `BL`, `W3`). Filtering strictly for `Series == 'EQ'` reduced the dataset to exactly 1,240 trading sessions per stock (**19,840 total rows**).
- **Duplicate Keys:** Exactly **0 duplicate `(Symbol, Date)` keys**.
- **Date Integrity:** Exactly **0 invalid dates**.
- **Price Bounds:** Exactly **0 non-positive closing prices**; **0 OHLC boundary violations** ($\text{High} \ge \max(\text{Open}, \text{Close})$ and $\text{Low} \le \min(\text{Open}, \text{Close})$ for 100% of rows).

### 4.2 Corporate Action Audit and Primary-Source Verification
Four mechanical corporate actions were confirmed via official NSE circulars and reconciled:

1. **MOTILALOFS (10 June 2024):** 3:1 Bonus Issue. Ratio = $0.2500$. Pre-event prices multiplied by $0.25$; volumes multiplied by $4.0$. Corrected raw drop of $-72.96\%$ to true adjusted market return of $+8.15\%$. (NSE Circular: `NSE/CML/2024/Bonus-MOFSL`).
2. **RELIANCE (28 October 2024):** 1:1 Bonus Issue. Ratio = $0.5000$. Pre-event prices multiplied by $0.50$; volumes multiplied by $2.0$. Corrected raw drop of $-49.76\%$ to true adjusted market return of $+0.49\%$. (NSE Circular: `NSE/CML/64512`).
3. **SIEMENS (07 April 2025):** Demerger of Energy Business into Siemens Energy India Ltd (SEIL). Special Pre-Open Session auction price discovery established continuing base price of INR 2,449.95 vs cum-close INR 4,928.15. Ratio = $2,449.95 / 4,928.15 \approx 0.49713$. Corrected raw drop of $-42.93\%$ to adjusted return of $+14.80\%$. (NSE Circular: `NSE/CML/2025/Demerger-SIEMENS`).
4. **HDFCBANK (26 August 2025):** 1:1 Bonus Issue. Ratio = $0.5000$. Pre-event prices multiplied by $0.50$; volumes multiplied by $2.0$. Corrected raw drop of $-50.44\%$ to adjusted return of $-0.88\%$. (NSE Circular: `NSE/CML/2025/Bonus-HDFCBANK`).

### 4.3 Verified Non-Corporate Action Large Moves
All 9 remaining large price moves (> $\pm 15\%$) were verified as genuine supply/demand price discovery:
- **ADANIGREEN (27 & 30 Jan 2023):** $-20\%$ lower circuits on Hindenburg Research report release.
- **ADANIGREEN (04–06 Dec 2023):** $+20\%$ upper circuit on Supreme Court hearing conclusions.
- **Systemic Election Crash (04 Jun 2024):** Market-wide plunge on Lok Sabha vote counting (ABB $-12.2\%$, ADANIGREEN $-19.2\%$, MOTILALOFS $-12.5\%$, SBIN $-14.4\%$, SIEMENS $-12.4\%$).
- **ADANIGREEN (21 & 29 Nov 2024):** $-18.9\%$ selloff on US DOJ indictment, followed by $+21.8\%$ rebound.
- **POWERINDIA (30 Jan 2025):** $+20\%$ upper circuit on Q3 FY25 financial blowout (+498% YoY profit).
- **YESBANK (07 Feb 2024):** $+17.3\%$ rally on Q3 turnaround (+349% YoY profit) and stake sale speculation.

---

## 5. Exploratory Data Analysis (EDA)

1. **Price Dispersion:** Traded prices span three orders of magnitude, from INR 12.10 (YESBANK) to INR 26,300.00 (POWERINDIA), underscoring the absolute necessity of normalizing raw prices into scale-free percentage returns and moving-average distance ratios.
2. **Volatility Clustering:** Daily return distributions exhibit classic fat-tailed leptokurtosis (kurtosis $> 6.5$) with pronounced volatility clustering during market shock periods (Hindenburg report in Jan 2023, Lok Sabha election counting in June 2024).
3. **Turnover Consistency:** Daily Rupee turnover ($\text{Price} \times \text{Volume}$) averages INR 180 crore across the universe, with liquid mega-caps (Reliance, HDFC Bank, ICICI Bank peers) providing consistent market depth.

---

## 6. Feature Engineering

Twenty predictive features were engineered strictly using information available at the close of trading session $t$:

| Feature Group | Feature Identifiers | Mathematical Definition | Historical Window |
| :--- | :--- | :--- | :--- |
| **Momentum Returns** | `return_1d`, `return_2d`, `return_5d`, `return_10d`, `return_20d` | $P_t / P_{t-k} - 1.0$ | $k \in \{1, 2, 5, 10, 20\}$ sessions |
| **Trend Distance Ratios** | `close_vs_sma_5`, `close_vs_sma_10`, `close_vs_sma_20`, `close_vs_sma_50` | $P_t / \text{SMA}_k(P) - 1.0$ | $k \in \{5, 10, 20, 50\}$ sessions |
| **Rolling Volatility** | `volatility_10d`, `volatility_20d` | $\text{Std}(\text{return}_{1d}, k)$ | $k \in \{10, 20\}$ sessions |
| **Intraday Structure** | `daily_range_pct`, `open_close_return` | $(H_t - L_t)/P_t$, $P_t/O_t - 1.0$ | Current session $t$ |
| **Volume Dynamics** | `volume_change_1d`, `volume_vs_avg20`, `trades_vs_avg20` | $V_t/V_{t-1}-1$, $V_t/\text{SMA}_{20}(V)-1$ | Past 20 sessions |
| **Delivery Quality** | `delivery_pct` | $(\text{Deliverable Qty}/\text{Traded Qty}) \times 100$ | Current session Bhavcopy |
| **Oscillators** | `rsi_14` | $100 - [100 / (1 + \text{RS}_{14})]$ | Wilder's 14-session EMA |
| **Calendar Signals** | `day_of_week`, `month` | Integer day (0–4) and month (1–12) | Current session calendar |

*Point-in-Time Proof:* For all historical dates outside an event-crossing window, the retrospective adjustment multiplier cancels out in the numerator and denominator ($\frac{P_t \cdot m}{P_{t-1} \cdot m} - 1 = \frac{P_t}{P_{t-1}} - 1$). The maximum empirical difference between adjusted and unadjusted features prior to ex-dates is $< 9.99 \times 10^{-17}$.

---

## 7. Target Construction

The primary prediction target represents the directional return 5 trading sessions ahead:
$$\text{future\_adj\_close\_5d}_{s, t} = \text{adj\_close}_{s, t+5}$$
$$\text{future\_return\_5d}_{s, t} = \frac{\text{adj\_close}_{s, t+5}}{\text{adj\_close}_{s, t}} - 1.0$$
$$\text{target\_up\_5d}_{s, t} = \begin{cases} 
1.0 & \text{if } \text{future\_adj\_close\_5d}_{s, t} > \text{adj\_close}_{s, t} \\
0.0 & \text{if } \text{future\_adj\_close\_5d}_{s, t} \le \text{adj\_close}_{s, t} \\
\text{NaN} & \text{if } t > T_s - 5
\end{cases}$$

- **Per-Symbol Shift Isolation:** Forward shifts are executed strictly within `groupby('Symbol')`. Zero cross-ticker contamination occurs.
- **Tie-Breaking Rule:** Exact ties ($P_{t+5} == P_t$; exactly 39 occurrences across 19,840 rows, or $0.20\%$) are assigned to Class `0.0`.
- **Target Distribution:** 10,328 Class 1s ($52.27\%$) and 9,432 Class 0s ($47.73\%$). Exactly 80 rows ($16 \times 5$) have unavailable targets at the terminal edge.

---

## 8. Models and Baselines

Five distinct classifier pipelines were evaluated:
1. **Dummy Baseline:** Mode predictor (`strategy="most_frequent"`). Predicts the historical majority class (Class 1) for every row.
2. **Logistic Regression:** L2 regularized ($C=0.1$, `class_weight="balanced"`, standardized inputs via `StandardScaler`).
3. **Random Forest:** Ensemble of 300 de-correlated decision trees (`max_depth=8`, `min_samples_leaf=15`, `max_features="sqrt"`, `class_weight="balanced"`).
4. **HistGradientBoosting:** Histogram-based gradient boosting trees (`max_iter=150`, `max_depth=5`, `min_samples_leaf=20`, `l2_regularization=1.5`, `learning_rate=0.04`).
5. **XGBoost:** Extreme Gradient Boosting (`n_estimators=150`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`).
6. **Secondary Regression Model:** `HistGradientBoostingRegressor` predicting continuous forward return ($\text{future\_return\_5d}$).

---

## 9. Chronological Train-Validation-Test Methodology

To prevent temporal leakage, all splits were strictly chronological:
- **Development Partition (85%):** 16,784 rows across 1,049 trading sessions (2021-10-25 to 2026-01-13).
- **Holdout Embargo / Purge Buffer:** 5 full trading sessions (2026-01-14 to 2026-01-21).
- **Untouched Holdout Partition (15%):** 2,896 rows across 181 trading sessions (2026-01-22 to 2026-10-15).
- **Walk-Forward Cross-Validation:** 3 expanding folds across development data with 5-session purge buffers between every train and validation window.

---

## 10. Evaluation Metrics and Actual Results

### 10.1 Walk-Forward Cross-Validation Results (Development Period)

| Model Architecture | Mean Out-of-Sample ROC-AUC | Out-of-Sample Accuracy | Balanced Accuracy | F1-Score | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | $0.5000 \pm 0.0000$ | $0.5303 \pm 0.0165$ | $0.5000 \pm 0.0000$ | $0.6930 \pm 0.0134$ | $0.2491 \pm 0.0008$ |
| **Logistic Regression** | $\mathbf{0.5159 \pm 0.0130}$ | $0.5092 \pm 0.0118$ | $\mathbf{0.5091 \pm 0.0115}$ | $0.5186 \pm 0.0388$ | $0.2520 \pm 0.0016$ |
| **Random Forest** | $0.5046 \pm 0.0157$ | $0.5071 \pm 0.0125$ | $0.5033 \pm 0.0143$ | $0.5516 \pm 0.0345$ | $0.2504 \pm 0.0011$ |
| **HistGradientBoosting** | $0.5044 \pm 0.0127$ | $0.5030 \pm 0.0112$ | $0.5002 \pm 0.0101$ | $0.5284 \pm 0.0326$ | $0.2525 \pm 0.0028$ |
| **XGBoost** | $0.5062 \pm 0.0200$ | $0.5177 \pm 0.0146$ | $0.5066 \pm 0.0175$ | $0.6277 \pm 0.0331$ | $0.2514 \pm 0.0021$ |

### 10.2 Untouched Holdout Benchmark Results (2,896 Rows; 2026-01-22 to 2026-10-15)

| Model Architecture | Holdout ROC-AUC | Holdout PR-AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.5000 | 0.4779 | 0.4779 | 0.5000 | 0.4779 | 1.0000 | 0.6467 | 0.5221 |
| **Logistic Regression** | 0.4642 | 0.4584 | 0.4706 | 0.4683 | 0.4427 | 0.4162 | 0.4291 | 0.2526 |
| **Random Forest** | 0.5017 | 0.4871 | 0.4914 | 0.4990 | 0.4772 | 0.6720 | 0.5581 | 0.2535 |
| **HistGradientBoosting** | $\mathbf{0.5187}$ | $\mathbf{0.4999}$ | $\mathbf{0.5180}$ | $\mathbf{0.5223}$ | $\mathbf{0.4965}$ | 0.6199 | 0.5514 | 0.2558 |
| **XGBoost** | 0.5033 | 0.4885 | 0.4845 | 0.4990 | 0.4773 | 0.8288 | 0.6058 | 0.2577 |

*Secondary Regression Experiment:* HistGradientBoostingRegressor achieved MAE = $0.0347$ (3.47%), RMSE = $0.0472$ (4.72%), and Directional Accuracy = $47.10\%$ ($R^2 \approx -0.01$).

---

## 11. Statistical Uncertainty and Significance Testing

### 11.1 Date-Level Moving-Block Bootstrap ($L=5$ Trading Dates, $B=2,000$ Replications)
To account for cross-sectional stock dependence and 5-session serial autocorrelation across overlapping forward returns, a moving-block bootstrap over trading dates was executed:

| Model Architecture | Holdout ROC-AUC Point | 95% Block Bootstrap CI | Superiority ($H_0: \text{AUC} \le 0.50$) $p$-value | Inferiority ($H_0: \text{AUC} \ge 0.50$) $p$-value | Two-Sided ($H_0: \text{AUC} = 0.50$) $p$-value | Conclusion ($\alpha = 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.5000 | $[0.5000, 0.5000]$ | $1.0000$ | $1.0000$ | $2.0000$ | Exact Baseline |
| **Logistic Regression** | 0.4642 | $[0.4160, 0.5070]$ | $0.9445$ | $0.0555$ | $0.1110$ | Fail to Reject $H_0$ (Not Significant) |
| **Random Forest** | 0.5017 | $[0.4513, 0.5536]$ | $0.4760$ | $0.5240$ | $0.9520$ | Fail to Reject $H_0$ (Indistinguishable from Chance) |
| **HistGradientBoosting** | 0.5187 | $[0.4783, 0.5586]$ | $0.1790$ | $0.8210$ | $0.3580$ | Fail to Reject $H_0$ (Edge is Not Significant) |
| **XGBoost** | 0.5033 | $[0.4573, 0.5503]$ | $0.4350$ | $0.5650$ | $0.8700$ | Fail to Reject $H_0$ (Indistinguishable from Chance) |

*Core Statistical Finding:* When temporal autocorrelation is properly preserved, the 95% confidence interval for HistGradientBoosting ($[0.4783, 0.5586]$) includes $0.50$ ($p = 0.1790$). No model demonstrates statistically significant predictive skill over chance.

---

## 12. Leakage and Data-Integrity Checks

1. **Temporal Horizon Isolation:** For the last development date (`2026-01-13`), the forward target evaluates on `2026-01-21`. The holdout partition begins on `2026-01-22`. Zero overlap occurs.
2. **In-Fold Preprocessing:** Scalers and imputers are strictly fitted on training partitions and applied out-of-sample.
3. **No Target-Derived Features:** Zero target, forward return, or future price columns appear in the feature matrix.
4. **Automated Test Suite:** 10 unit and integration tests executed in `pytest` with a 100% pass rate.

---

## 13. Limitations and Ethical Considerations

1. **Weak-Form Market Efficiency:** Liquid equities rapidly incorporate historical price data. Short-term daily technical indicators have near-zero marginal signal.
2. **Model Risk in Capital Deployment:** Deploying these models in live financial trading would result in negative expected value due to execution slippage, bid-ask spreads, and transaction costs.
3. **Siemens Demerger Volume Limitation:** Pre-demerger volume was scaled by $2.0115$ to maintain Rupee turnover invariance. Users must recognize this as a liquidity proxy rather than physical share volume.

---

## 14. Conclusion and Future Work

This research project delivers a fully audited, reproducible, and methodologically defensible quantitative machine learning system. By eliminating look-ahead leakage, reconciling corporate actions, and applying rigorous block-bootstrap uncertainty estimation, the project demonstrates that short-term equity directional prediction from daily price data yields performance close to chance ($0.51 - 0.52$ ROC-AUC).

### Future Directions:
1. **Alternative Data Integration:** Incorporate limit order book imbalances, institutional positioning, and corporate earnings surprises.
2. **Non-Uniform Temporal Horizons:** Evaluate 1-day intraday momentum and 20-day monthly trend persistence.
3. **Asset-Specific Clustered Modeling:** Train separate estimators for distinct sectoral regimes (IT, Banking, Industrials).

---

## 15. Reproduction Instructions

To reproduce all results, metrics, charts, and tests:

```bash
# 1. Navigate to model directory
cd "D:\B.Tech\Sem3\IML\Project\Stock_Prediction\model_development"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute master pipeline (Ingestion, Audit, Training, Holdout, Figures, Tests)
python run_project.py

# 4. Run automated test suite
pytest tests/ -v

# 5. Generate forecasts on latest completed market data
python predict.py --symbol RELIANCE
python predict.py --all
```
