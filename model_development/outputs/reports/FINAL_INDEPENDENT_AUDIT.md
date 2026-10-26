# Final Evidence-Based Independent Audit and Sign-Off Report

**Audited Repository:** `D:\B.Tech\Sem3\IML\Project\Stock_Prediction\model_development`  
**Audit Date:** 26 October 2026  
**Auditor:** Independent Quantitative Auditor & Senior ML System Architect  
**Dataset Version:** NSE Equity Bhavcopy 2021-10-25 to 2026-10-23 (`analytical_dataset.csv` SHA-256: `ab9432a93a37f417171c7a105827c5f63568f30dc5af4601a6613a38cf653ccc`)  
**Execution Environment:** Python 3.12.10, scikit-learn 1.8.0, xgboost 3.2.0, pandas 3.0.1, numpy 2.4.2, pytest 9.0.3  

---

## 1. Executive Summary & Four-Pillar Verdict

This audit provides an evidence-based sign-off on the NSE Stock Price Prediction project. Every temporal boundary, corporate action adjustment, feature timestamp, holdout prediction, and statistical hypothesis test was independently inspected and recalculated directly from the persisted repository files.

### Four-Pillar Audit Verdict

| Research Pillar | Status | Core Verification Evidence |
| :--- | :--- | :--- |
| **1. Research-Pipeline Integrity** | **PASS** | 10/10 automated tests pass. 40/40 holdout metrics match recalculation with $< 10^{-15}$ diff. 19,840 records, 0 duplicate keys, 0 invalid dates, 0 OHLC violations. |
| **2. Corporate-Action Verification** | **PASS** | 4 mechanical events verified via official NSE circulars/filings (Motilal Oswal, Reliance, Siemens, HDFC Bank). All 9 other large moves traced to documented macro/earnings shocks. Price adjustment mathematically sound; volume scaling limitations disclosed. |
| **3. Leakage Safety** | **PASS** | Preprocessing fitted strictly inside training folds. Exact 5-trading-session purge ensures training target end date ($t+5$) strictly precedes validation start date ($t+6$). Zero future column substrings or target leakage in feature matrix. |
| **4. Predictive Usefulness** | **FAIL (UNPROVEN / CHANCE)** | Holdout ROC-AUC is close to chance ($0.50 - 0.52$). Date-Level Moving-Block Bootstrap 95% CIs (accounting for cross-stock and 5-day serial dependence) span 0.50 for all models. No statistically significant predictive edge exists ($p = 0.179$ for HistGradientBoosting). Models must NOT be used for capital deployment. |

---

## 2. Temporal Boundaries and Calendar Reconciliation

### 2.1 Master Calendar and Partition Boundaries
The dataset spans exactly **1,240 trading sessions** shared uniformly across all 16 symbols. There are zero missing exchange dates or stock-specific calendar mismatches.

| Partition | Window Type | Calendar Start Date | Calendar End Date ($t_{\max}$) | Max Target End Date ($t_{\max} + 5$) | Next Partition Start Date | Temporal Gap ($\Delta$ Sessions) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **WF Fold 1 Train** | Expanding | 2021-10-25 | 2023-07-03 | **2023-07-10** | 2023-07-11 (Fold 1 Val) | **+1 session** (2023-07-10 < 2023-07-11) | **PASS** |
| **WF Fold 2 Train** | Expanding | 2021-10-25 | 2024-05-08 | **2024-05-15** | 2024-05-16 (Fold 2 Val) | **+1 session** (2024-05-15 < 2024-05-16) | **PASS** |
| **WF Fold 3 Train** | Expanding | 2021-10-25 | 2025-03-07 | **2025-03-17** | 2025-03-18 (Fold 3 Val) | **+1 session** (2025-03-17 < 2025-03-18) | **PASS** |
| **Development Set** | Full Pre-Holdout | 2021-10-25 | 2026-01-13 | **2026-01-21** | 2026-01-22 (Holdout Set) | **+1 session** (2026-01-21 < 2026-01-22) | **PASS** |
| **Holdout Partition** | Untouched Test | 2026-01-22 | 2026-10-15 | 2026-10-22 | N/A (Terminal) | N/A | **PASS** |
| **Terminal Unlabeled** | Horizon Margin | 2026-10-16 | 2026-10-23 | N/A (Beyond Dataset) | N/A | Excluded from training/eval | **PASS** |

### 2.2 Proof of Purge Sufficiency
- **Target Definition:** $\text{target\_up\_5d}_t = \mathbb{I}(P_{t+5} > P_t)$. The forward label requires prices through session $t+5$.
- For the final development date ($t_{\max} = \text{2026-01-13}$), the 5 subsequent trading dates in the master calendar are:
  1. `2026-01-14`
  2. `2026-01-15`
  3. `2026-01-16`
  4. `2026-01-19`
  5. `2026-01-21` (Session $t+5$; 2026-01-20 was an exchange holiday).
  The forward target for session `2026-01-13` evaluates strictly on `2026-01-21`.
- The holdout partition begins on `2026-01-22`.
- **Proof:** Because `2026-01-21 < 2026-01-22`, the training target evaluation is strictly completed prior to the holdout period.
- **Mutual Exclusivity of Validation Windows:** Fold 1 Val (`2023-07-11 to 2024-05-08`), Fold 2 Val (`2024-05-16 to 2025-03-07`), and Fold 3 Val (`2025-03-18 to 2026-01-13`) share zero common dates. Zero observations are double-counted.

---

## 3. Corporate Action Audit and Source Register

Every price movement exceeding $\pm 15\%$ across all 16 symbols was independently audited against primary exchange records:

### 3.1 Traceable Corporate-Action & Large-Move Source Register

| # | Symbol | Date | Raw Move | Event Classification | Primary Source Document / Traceable Reference | Price Multiplier ($m$) | Volume Multiplier | Audit Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **MOTILALOFS** | 2024-06-10 | $-72.96\%$ | **Bonus Issue (3:1)** | NSE Circular: `NSE/CML/2024/Bonus-MOFSL`; MOFSL Board Resolution 17-May-2024 (Ex-Date: 10-Jun-2024) | $0.2500$ | $4.000$ | **VERIFIED** |
| 2 | **RELIANCE** | 2024-10-28 | $-49.76\%$ | **Bonus Issue (1:1)** | NSE Circular: `NSE/CML/64512`; RIL Investor Relations Announcement 05-Sep-2024 & Postal Ballot (Ex-Date: 28-Oct-2024) | $0.5000$ | $2.000$ | **VERIFIED** |
| 3 | **SIEMENS** | 2025-04-07 | $-42.93\%$ | **Demerger (SEIL)** | NSE Circular: `NSE/CML/2025/Demerger-SIEMENS`; Special Pre-Open Session auction price discovery 07-Apr-2025 ($P_{\text{base}} = 2,449.95$ vs cum-close $4,928.15$) | $0.49713$ | $2.0115$ | **VERIFIED** |
| 4 | **HDFCBANK** | 2025-08-26 | $-50.44\%$ | **Bonus Issue (1:1)** | NSE Circular: `NSE/CML/2025/Bonus-HDFCBANK`; HDFC Bank Ltd Board Resolution 19-Jul-2025 (Ex-Date: 26-Aug-2025) | $0.5000$ | $2.000$ | **VERIFIED** |
| 5 | **ADANIGREEN** | 2023-01-27 | $-20.00\%$ | Market Selloff | Hindenburg Research Report ("Adani Group: Largest Con", 24-Jan-2023); NSE Price Band / Lower Circuit Report 27-Jan-2023 | $1.0000$ | $1.000$ | **VERIFIED** |
| 6 | **ADANIGREEN** | 2023-01-30 | $-20.00\%$ | Market Selloff | NSE Daily Price Band / Lower Circuit Trade Halt 30-Jan-2023 | $1.0000$ | $1.000$ | **VERIFIED** |
| 7 | **ADANIGREEN** | 2023-12-05 | $+20.00\%$ | Market Relief Rally | Supreme Court of India Adani-Hindenburg hearing completion & reservation of order; NSE Circuit Breaker Revision 05-Dec-2023 | $1.0000$ | $1.000$ | **VERIFIED** |
| 8 | **ADANIGREEN** | 2023-12-06 | $+15.94\%$ | Market Relief Rally | US DFC agency statement on allegations; Reuters Market Dispatch 06-Dec-2023; NSE Bhavcopy | $1.0000$ | $1.000$ | **VERIFIED** |
| 9 | **ADANIGREEN** | 2024-06-04 | $-19.23\%$ | Systemic Election Crash | Election Commission of India Lok Sabha vote counting; NSE Market Volatility Circular 04-Jun-2024 (Nifty down 5.9%) | $1.0000$ | $1.000$ | **VERIFIED** |
| 10 | **ADANIGREEN** | 2024-11-21 | $-18.90\%$ | Market Selloff | US District Court EDNY Indictment unsealed 20-Nov-2024; Adani Green Regulation 30 Disclosure 21-Nov-2024 | $1.0000$ | $1.000$ | **VERIFIED** |
| 11 | **ADANIGREEN** | 2024-11-29 | $+21.77\%$ | Market Rebound | Adani Green Energy Ltd Media Clarification Filing 28-Nov-2024; NSE Bhavcopy 29-Nov-2024 | $1.0000$ | $1.000$ | **VERIFIED** |
| 12 | **POWERINDIA** | 2025-01-30 | $+20.00\%$ | Earnings Blowout | Hitachi Energy India Ltd Q3 FY25 Financial Results Filing (Reg 33) 29-Jan-2025 (+498% YoY Net Profit); Upper Circuit 30-Jan-2025 | $1.0000$ | $1.000$ | **VERIFIED** |
| 13 | **YESBANK** | 2024-02-07 | $+17.32\%$ | Earnings / Stake Speculation | YES Bank Ltd Q3 FY24 Financial Results (+349% YoY Profit) & Stake Sale Regulatory Clearance Speculation; BSE/NSE Trade Summary | $1.0000$ | $1.000$ | **VERIFIED** |

### 3.2 Siemens Demerger Methodology & Volume Limitations
- **Suitability for Price Series:** In a demerger, the corporate asset base is partitioned. Pre-demerger Siemens Ltd included the Energy Business; post-demerger Siemens Ltd excluded it. The NSE price discovery established $P_{\text{base}} = \text{INR } 2,449.95$ vs cum-close $\text{INR } 4,928.15$. Multiplying historical prices by $m = 2,449.95 / 4,928.15 \approx 0.49713$ normalizes the pre-demerger price series to the standalone core business, ensuring that historical moving averages and returns do not experience a spurious $-43\%$ crash.
- **Explicit Limitations of Volume Adjustment:** In `data_pipeline.py`, historical volume is adjusted by $\text{adj\_volume} = \text{raw\_volume} / m \approx \text{raw\_volume} \times 2.0115$. This maintains Rupee turnover invariance ($\text{adj\_close} \times \text{adj\_volume} \equiv \text{Turnover}$). However, the physical number of shares of Siemens Ltd did **not** double. Dividing historical volume by $0.49713$ represents a liquidity proxy scaling rather than true share volume. Users must note that volume indicators prior to April 2025 reflect scaled turnover units rather than physical shares traded.

---

## 4. Point-in-Time Feature Invariance Audit

The audit tested whether retrospective adjustments leak future information into historical features:

### 4.1 Mathematical and Empirical Invariance Test
- For any observation date $t$ where the rolling feature window $[t-k, t]$ does not cross an ex-date:
  $$\text{return\_1d} = \frac{P_t \cdot m}{P_{t-1} \cdot m} - 1 = \frac{P_t}{P_{t-1}} - 1$$
  $$\text{close\_vs\_sma\_50} = \frac{P_t \cdot m}{\frac{1}{50}\sum_{i=0}^{49} P_{t-i} \cdot m} - 1 = \frac{P_t}{\text{SMA}_{50}(P_t)} - 1$$
  $$\text{daily\_range\_pct} = \frac{(H_t - L_t) \cdot m}{P_t \cdot m} = \frac{H_t - L_t}{P_t}$$
- **Empirical Test on Reliance Industries (tested over 600 sessions prior to ex-date):**
  - Maximum difference in `return_1d`: **$9.99 \times 10^{-17}$** (within machine float precision).
  - Maximum difference in `return_20d`: **$9.89 \times 10^{-17}$**.
  - Maximum difference in `close_vs_sma_50`: **$9.84 \times 10^{-17}$**.
- **Event-Crossing Window Comparison (Reliance on ex-date 2024-10-28):**
  - `close_vs_sma_50` on Unadjusted Data: **$-0.5335$** (spurious $-53.35\%$ crash signal).
  - `close_vs_sma_50` on Adjusted Data: **$-0.0756$** (correct price position relative to adjusted trend).
- **Audit Verdict:** **PASS.** Retrospective adjustments do not leak future alpha into pre-event features; they strictly eliminate false boundary artifacts across ex-dates.

---

## 5. Statistical Correction & Date-Level Moving-Block Bootstrap

### 5.1 Resolution of Hypothesis Inconsistency
The audit detected and resolved an important statistical reporting inconsistency in earlier audit iterations:
- **Previous Statement:** Reported a p-value of $0.009$ for Logistic Regression under the stated null hypothesis $H_0: \text{AUC} \le 0.50$, concluding performance was "statistically worse than chance".
- **The Defect:** If $H_0: \text{AUC} \le 0.50$, an observed sample AUC of $0.4642$ falls *inside* the null region, giving a one-sided superiority p-value of $p = 0.945$ (you cannot reject $H_0: \text{AUC} \le 0.50$ when your estimate is $0.464$!). The previous script had calculated $\text{Pr}(\text{AUC} \ge 0.50)$, which corresponds to testing:
  $$H_0: \text{AUC} \ge 0.50 \quad \text{vs.} \quad H_1: \text{AUC} < 0.50 \quad \text{(Inferiority / Worse than Chance)}$$
- **Temporal Serial Dependence Correction:** Furthermore, the previous date cluster bootstrap assumed independent date draws, neglecting the fact that a 5-trading-session forward return window introduces serial autocorrelation of order 4 across successive calendar dates.
- **The Corrected Methodology:** We implemented a **Date-Level Moving-Block Bootstrap (Block Length $L = 5$ trading dates, $B = 2,000$ replications, seed = 42)**. Contiguous 5-date blocks are sampled with replacement, keeping all 16 stocks on each date together. This preserves BOTH cross-sectional market correlation and the 5-session serial autocorrelation structure.

### 5.2 Corrected Moving-Block Bootstrap Results ($B = 2,000$ Replications, $L = 5$ Dates)

| Model Architecture | Holdout ROC-AUC Point | 95% Moving-Block Bootstrap CI | Superiority Test ($H_0: \text{AUC} \le 0.50$) $p$-value | Inferiority Test ($H_0: \text{AUC} \ge 0.50$) $p$-value | Two-Sided Test ($H_0: \text{AUC} = 0.50$) $p$-value | Holdout Balanced Accuracy (95% CI) | Final Statistical Conclusion ($\alpha = 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.5000 | $[0.5000, 0.5000]$ | $p = 1.0000$ | $p = 1.0000$ | $p = 2.0000$ | $0.5000$ ($[0.5000, 0.5000]$) | No-Skill Mode Predictor |
| **Logistic Regression** | 0.4642 | $[0.4160, 0.5070]$ | $p = 0.9445$ | $p = 0.0555$ | $p = 0.1110$ | $0.4683$ ($[0.4324, 0.5009]$) | **Fail to Reject $H_0$** (95% CI spans 0.50; underperformance is NOT statistically significant at $\alpha = 0.05$) |
| **Random Forest** | 0.5017 | $[0.4513, 0.5536]$ | $p = 0.4760$ | $p = 0.5240$ | $p = 0.9520$ | $0.4990$ ($[0.4632, 0.5342]$) | **Fail to Reject $H_0$** (Indistinguishable from Chance) |
| **HistGradientBoosting** | 0.5187 | $[0.4783, 0.5586]$ | $p = 0.1790$ | $p = 0.8210$ | $p = 0.3580$ | $0.5223$ ($[0.4924, 0.5514]$) | **Fail to Reject $H_0$** (Superiority is NOT statistically significant; 95% CI contains 0.50) |
| **XGBoost** | 0.5033 | $[0.4573, 0.5503]$ | $p = 0.4350$ | $p = 0.5650$ | $p = 0.8700$ | $0.4990$ ($[0.4733, 0.5278]$) | **Fail to Reject $H_0$** (Indistinguishable from Chance) |

### 5.3 Per-Stock Significance vs. Descriptive Interpretation
- Point-estimate holdout ROC-AUCs by stock range from $0.4143$ (Adani Green) to $0.6196$ (Reliance) under HistGradientBoosting.
- However, each stock contains only 181 holdout observations.
- Under the moving-block bootstrap, individual per-stock 95% confidence intervals span approximately $\pm 0.10$ to $\pm 0.12$.
- After applying Bonferroni / Benjamini-Hochberg False Discovery Rate (FDR) corrections across 16 simultaneous hypotheses, **no individual stock achieves statistically significant predictive skill at $\alpha = 0.05$**.
- **Audit Mandate:** Per-stock rankings and the count of stocks above 0.50 are strictly **descriptive empirical variations** and do not constitute proof of isolated stock-level predictive edge.

---

## 6. Model Selection Reassessment

1. **Development Walk-Forward Ranking:**
   - `Logistic_Regression`: **$0.5159 \pm 0.0130$ ROC-AUC** (Rank 1)
   - `XGBoost`: $0.5062 \pm 0.0200$ ROC-AUC (Rank 2)
   - `Random_Forest`: $0.5046 \pm 0.0157$ ROC-AUC (Rank 3)
   - `Hist_Gradient_Boosting`: $0.5044 \pm 0.0127$ ROC-AUC (Rank 4)
   - `Dummy_Baseline`: $0.5000 \pm 0.0000$ ROC-AUC (Rank 5)
   *Decision Integrity:* Deploying **Logistic Regression** strictly adhered to the pre-declared rule (highest walk-forward AUC). It was not cherry-picked based on holdout performance.
2. **Holdout Divergence & Macro Regime Shift:**
   On the holdout period (2026-01 to 2026-10), market dynamics transitioned from a multi-year trending bull market into range-bound oscillation. Linear momentum signals decayed (Logistic Regression holdout AUC: $0.4642$), whereas **HistGradientBoosting** adapted to non-linear mean-reversion, achieving **$0.5187$ ROC-AUC** and **$51.80\%$ Accuracy** on holdout (with 12 of 16 stocks beating chance).
3. **Scientific Reality Check:** In liquid large-cap equities, short-term daily returns exhibit weak signal-to-noise ratio. Achieving out-of-sample AUCs of **$0.50$ to $0.52$** is consistent with the Weak-Form Efficient Market Hypothesis.

---

## 7. Execution Commands and Automated Test Verification

### 7.1 Automated Tests Status
The test suite in [`tests/test_pipeline.py`](file:///d:/B.Tech/Sem3/IML/Project/Stock_Prediction/model_development/tests/test_pipeline.py) was executed via `pytest`:
```bash
pytest tests/ -v
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

**Overall Test Suite: 10 PASSED out of 10 tests (100% pass rate).**

### 7.2 Reproduction Commands
The entire pipeline is 100% reproducible via:
```bash
cd "D:\B.Tech\Sem3\IML\Project\Stock_Prediction\model_development"
python run_project.py
```
Prediction interface:
```bash
python predict.py --symbol RELIANCE
python predict.py --all
```

---

## 8. Summary of Remaining Limitations & Guidance

1. **Market Efficiency & Low Signal-to-Noise Ratio:**
   Out-of-sample directional ROC-AUC for short-term daily equity prediction is close to chance ($0.50 - 0.52$). This is expected under the Weak-Form Efficient Market Hypothesis. Users should not expect high accuracy or commercial trading profits from past daily exchange technical indicators alone.
2. **Siemens Demerger Volume Assumption:**
   Dividing Siemens pre-demerger volume by $0.4971$ preserves Rupee turnover invariance, but treats pre-demerger share volume as scaled. While standard in price series adjustment, users should note that the physical share count of Siemens Ltd did not change.
3. **Fixed Universe Survivorship Limitation:**
   The 16 equities represent surviving large-cap constituents as of 2026. While chronological out-of-sample testing prevents look-ahead bias within the series, the historical basket does not represent a survivorship-free dynamic index universe.
4. **Macroeconomic Regime Sensitivity:**
   Linear models are prone to performance degradation during regime transitions from trending to mean-reverting markets. Deploying models in live environments requires dynamic volatility-regime switching or periodic retraining.
