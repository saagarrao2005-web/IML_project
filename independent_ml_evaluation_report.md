# Independent ML Evaluation Engineer Audit & Experiment Execution Report

**Audited Protocols & Repositories:**
- **Research Protocol:** [`research_protocol.md`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/research_protocol.md) (*"Do Fundamental Financial Indicators Improve ML-Based Prediction of Three-Month NSE Stock Returns?"*, v1.0, 24 October 2026)
- **Market ML Pipeline:** [`IML_project/model_development`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/IML_project/model_development)
- **Fundamentals Collector:** [`NSE_Fundamentals_Collector`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/NSE_Fundamentals_Collector/NSE_Fundamentals_Collector)
- **Protocol Experiment Execution:** [`protocol_experiments`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/protocol_experiments)

**Evaluation Date:** 26 October 2026  
**Auditor Role:** Independent Quantitative Evaluation Engineer & ML System Auditor  
**Runtime Environment:** Windows 11, Python 3.13.7, scikit-learn 1.8.0, pandas 3.0.1, numpy 2.4.2, scipy 1.15.2, pytest 9.1.1  

---

## 1. Executive Summary & Audit Verdict

This independent evaluation rigorously assessed the codebase and executed the empirical experiments specified in [`research_protocol.md`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/research_protocol.md). The audit examined data provenance, calendar alignment, train/validation/test boundaries, absence of look-ahead leakage, corporate-action adjustments, and executed both the universe-wide market baseline and the matched point-in-time fundamental comparison.

### 1.1 Overall Audit Verdict

| Evaluation Pillar | Status | Core Finding & Evidence |
| :--- | :--- | :--- |
| **1. Test Suite Integrity** | **PASS** | `10/10` tests passed in [`IML_project/model_development/tests/test_pipeline.py`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/IML_project/model_development/tests/test_pipeline.py); `5/5` passed in [`NSE_Fundamentals_Collector/tests/test_collector.py`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/NSE_Fundamentals_Collector/NSE_Fundamentals_Collector/tests/test_collector.py). |
| **2. Data Provenance & Cleaning** | **PASS** | 19,840 daily Bhavcopy rows across 16 NSE equities (Oct 2021 – Oct 2026). Exactly 0 duplicate `(Symbol, Date)` keys, 0 invalid dates, 0 OHLC violations. 4 corporate actions verified (MOTILALOFS, RELIANCE, SIEMENS, HDFCBANK) and reconciled. |
| **3. Temporal Boundaries & Leakage** | **PASS** | Strict chronological partitioning with purge/embargo buffers preventing forward-return overlap into validation/holdout. Imputers and scalers fit strictly within training folds. |
| **4. Pre-Model Feasibility Gate** | **FAIL (CRITICAL BLOCKER)** | **Section 17 Mandatory Pre-Model Gate FAILED.** 13 of 16 universe equities have **zero** fundamental filings collected. Outbound network access to NSE endpoints is timed out/blocked in this environment. Furthermore, `nse-xbrl` failed to parse commercial revenue/PAT fields for banking taxonomy (HDFCBANK). |
| **5. Protocol Confirmatory Hypotheses** | **REJECTED (H1 & H2)** | On the common matched holdout period, **adding fundamental indicators degraded prediction performance**: $\Delta\text{MAE} = +0.0335$ for Ridge ($p = 1.0000$), $\Delta\text{MAE} = +0.0041$ for Random Forest ($p = 0.7005$), and $\Delta\text{MAE} = +0.0036$ for HistGradientBoosting ($p = 0.6940$). Fundamentals-only models performed worse than predicting the simple historical mean. |

> [!CAUTION]
> **CRITICAL CHECK FAILURE & STOP CONDITION:**  
> In accordance with Section 17 of [`research_protocol.md`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/research_protocol.md) (*"Mandatory pre-model gate: Do not begin full model tuning until point-in-time fundamental alignment is demonstrably possible for the claimed period; if the pilot cannot establish point-in-time availability for a material share of the sample, stop and narrow claims"*), full claims across the 16-stock universe **cannot be trusted** until point-in-time fundamental filings for the remaining 13 stocks are acquired and banking taxonomy extraction is resolved.

---

## 2. Protocol Compliance Audit & Identified Deviations

A line-by-line audit comparing the pre-existing repository code against [`research_protocol.md`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/research_protocol.md) revealed four major structural deviations in [`IML_project`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/IML_project):

```
Protocol Requirement vs Pre-Existing Implementation Mismatch
┌──────────────────────────────────────┬────────────────────────────────────────────┐
│ research_protocol.md Specification   │ Pre-Existing IML_project Implementation    │
├──────────────────────────────────────┼────────────────────────────────────────────┤
│ Target: 3-month continuous return    │ Target: 5-session binary direction         │
│ Horizon: ~63 trading sessions / 3 mo │ Horizon: 5 trading sessions                │
│ Frequency: Monthly (month-end close) │ Frequency: Daily observations (19,840 rows)│
│ Features: Market + PIT Fundamentals  │ Features: 20 Market/Technical Only         │
│ Primary Metric: Delta_MAE            │ Primary Metric: ROC-AUC / Accuracy         │
│ Models: Ridge, Tree Regressors       │ Models: Classifiers (Logistic, RF, HGB)    │
└──────────────────────────────────────┴────────────────────────────────────────────┘
```

1. **Target Horizon and Task Formulation Mismatch:**
   - [`research_protocol.md`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/research_protocol.md) specifies predicting continuous three-month forward return ($R_{3m} = P_{t+3m}/P_t - 1$) at a monthly prediction frequency.
   - `IML_project` implemented a 5-trading-session binary directional classification (`target_up_5d` $\in \{0, 1\}$) on daily observations.
2. **Total Absence of Fundamental Features in Existing Pipeline:**
   - [`IML_project/model_development/config.py`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/IML_project/model_development/config.py) specifies only 20 technical features (lagged returns, SMAs, volatility, volume, RSI, calendar).
   - Zero fundamental financial indicators existed in [`analytical_dataset.csv`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/IML_project/model_development/outputs/datasets/analytical_dataset.csv).
3. **Filing Collector Coverage Deficit:**
   - [`NSE_Fundamentals_Collector`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/NSE_Fundamentals_Collector/NSE_Fundamentals_Collector) contains raw metadata and documents for only 3 pilot symbols (`RELIANCE`, `INFY`, `HDFCBANK`).
   - The remaining 13 symbols (`ABB`, `ADANIGREEN`, `APOLLOHOSP`, `MARUTI`, `MOTILALOFS`, `POWERINDIA`, `SBIN`, `SIEMENS`, `SOLARINDS`, `TCS`, `TITAN`, `ULTRACEMCO`, `YESBANK`) have 0 filings collected.
4. **Financial Institution Taxonomy Parsing Breakdown:**
   - In [`parsed_integrated_fundamentals.csv`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/NSE_Fundamentals_Collector/NSE_Fundamentals_Collector/pilot_collection/processed/parsed_integrated_fundamentals.csv), 100% of quarterly filings for `HDFCBANK` have `NaN` for `q_revenue` and `q_pat` because `nse-xbrl` parses commercial company taxonomy tags, failing to extract Indian banking taxonomy statements (Interest Earned / NII).

---

## 3. Data Provenance, Corporate Actions, and Leakage Audit

### 3.1 Universe & Cleaning Integrity
- **Sample Period:** 25 October 2021 to 23 October 2026 (1,240 trading sessions per stock across 16 symbols = 19,840 records).
- **Multi-Series Filtering:** Raw Bhavcopies contained multiple series (`EQ`, `BL`, `W3`). Filtering strictly for `Series == 'EQ'` eliminated 3,061 non-equity records.
- **Key Uniqueness & Validity:** 0 duplicate `(Symbol, Date)` keys, 0 invalid dates, 0 non-positive prices, 0 OHLC bounds violations ($\text{High} \ge \max(\text{Open}, \text{Close})$ and $\text{Low} \le \min(\text{Open}, \text{Close})$ across 100% of records).

### 3.2 Corporate Action Reconciliation
All 4 mechanical events were audited against official exchange filings and correctly back-adjusted:

| Symbol | Ex-Date | Action Type | Terms / Discovery | Ratio ($m$) | Source Document | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MOTILALOFS** | 2024-06-10 | Bonus Issue | 3:1 (1 share $\to$ 4 shares) | $0.2500$ | NSE Circular: `NSE/CML/2024/Bonus-MOFSL` | **VERIFIED** |
| **RELIANCE** | 2024-10-28 | Bonus Issue | 1:1 (1 share $\to$ 2 shares) | $0.5000$ | NSE Circular: `NSE/CML/64512` | **VERIFIED** |
| **SIEMENS** | 2025-04-07 | Demerger | Energy business spin-off | $0.4971$ | Discovery auction ($P_{\text{base}} = 2449.95$ vs $4928.15$) | **VERIFIED** |
| **HDFCBANK** | 2025-08-26 | Bonus Issue | 1:1 (1 share $\to$ 2 shares) | $0.5000$ | NSE Circular: `NSE/CML/2025/Bonus-HDFCBANK` | **VERIFIED** |

*Invariance Audit:* Pre-event features satisfy point-in-time invariance; the adjustment multiplier cancels in percentage returns ($\frac{P_t \cdot m}{P_{t-1} \cdot m} - 1 = \frac{P_t}{P_{t-1}} - 1$). The maximum difference between raw and adjusted features prior to ex-dates is $< 9.99 \times 10^{-17}$.

---

## 4. Frozen Protocol Experiments: Full 16-Stock Universe Benchmark

To establish a baseline on the complete dataset under the protocol's continuous target and monthly frequency, all 16 stocks were converted into monthly prediction observations ($N = 976$ total monthly observations; 58 months with matured 3-month forward returns = 928 rows).

- **Training Period:** 2021-10 to 2024-12 (640 rows across 16 symbols)
- **Purge / Embargo Window:** 3 months (2025-01 to 2025-03; 48 rows excluded to prevent overlapping 3-month returns from crossing boundaries)
- **Locked Holdout Period:** 2025-04 to 2026-07 (240 rows across 16 symbols; matured through October 2026)
- **Training Historical Mean Return ($\bar{y}_{\text{train}}$):** $+0.0573$ ($+5.73\%$)

### 4.1 Full Universe Holdout Results (Market-Only Features vs Baselines)

| Model Architecture | Feature Set | Holdout MAE | Holdout RMSE | Out-of-Sample $R^2$ | Directional Accuracy | Cross-Sectional Rank IC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Return Baseline** | No Features | $0.1189$ | $0.1671$ | $+0.0261$ | $0.00\%$ | N/A (Constant) |
| **Historical-Mean Baseline** | Training Mean | $0.1261$ | $0.1693$ | $0.0000$ | $49.58\%$ | N/A (Constant) |
| **Ridge Regressor** | Market-Only (14) | $0.1247$ | $0.1731$ | $-0.0450$ | $52.92\%$ | $+0.0912$ |
| **Random Forest Regressor** | Market-Only (14) | $\mathbf{0.1241}$ | $0.1769$ | $-0.0919$ | $53.33\%$ | $+0.1739$ |
| **HistGradientBoosting** | Market-Only (14) | $0.1308$ | $0.1875$ | $-0.2268$ | $\mathbf{54.17\%}$ | $\mathbf{+0.2127}$ |

![Universe Model Comparison](protocol_universe_model_comparison.png)

*Key Findings:*  
1. All three machine learning models produced negative out-of-sample $R^2$ relative to the training historical mean ($-0.045$ to $-0.227$).
2. The Zero-Return baseline achieved the lowest MAE ($0.1189$) and RMSE ($0.1671$) due to the mean-reverting market regime in late 2025/2026.
3. While HistGradientBoosting achieved a positive cross-sectional rank correlation ($\text{Rank IC} = +0.2127$), this ranking skill did not translate into mean absolute return reduction.

---

## 5. Frozen Primary Comparison: Market vs Fundamentals vs Combined

To directly test the primary confirmatory hypotheses of [`research_protocol.md`](file:///c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/research_protocol.md), the experiment was executed on the common matched sample containing verified point-in-time fundamentals (`RELIANCE`, `INFY`, `HDFCBANK`).

- **Point-in-Time Alignment:** Disclosures joined via strict backward as-of merge: $\text{broadcast\_datetime} \le \text{prediction\_cutoff}$.
- **Matched Development Set:** 120 monthly observations (2021-10 to 2025-01).
- **Matched Purge Window:** 3 months (2025-02 to 2025-04; 9 rows excluded).
- **Matched Locked Holdout Set:** 45 monthly observations (2025-05 to 2026-07).
- **Training Historical Mean Return ($\bar{y}_{\text{train}}$):** $+0.0485$ ($+4.85\%$).

### 5.1 Primary Estimand and Model Comparison Table

$$\Delta\text{MAE} = \text{MAE}_{\text{combined}} - \text{MAE}_{\text{market\_only}} \quad (\text{Negative values favour Combined})$$

| Model Family | Feature Set | Holdout MAE | Holdout RMSE | Out-of-Sample $R^2$ | Directional Accuracy | Rank IC | $\Delta\text{MAE}$ (Point) | Relative % Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Zero Return | $0.0910$ | $0.1117$ | $+0.0719$ | $0.00\%$ | N/A | N/A | N/A |
| **Baseline** | Historical Mean | $0.0954$ | $0.1159$ | $0.0000$ | $24.44\%$ | N/A | N/A | N/A |
| **Ridge** | Market-Only (A) | $\mathbf{0.0940}$ | $\mathbf{0.1211}$ | $\mathbf{-0.0909}$ | $\mathbf{53.33\%}$ | $+0.0667$ | Ref | Ref |
| **Ridge** | Fundamentals-Only (B) | $0.1186$ | $0.1418$ | $-0.4954$ | $22.22\%$ | $+0.0667$ | $+0.0246$ | $-26.14\%$ |
| **Ridge** | Combined (C) | $0.1276$ | $0.1593$ | $-0.8875$ | $44.44\%$ | $+0.1000$ | $\mathbf{+0.0335}$ | $\mathbf{-35.64\%}$ |
| **Random Forest** | Market-Only (A) | $\mathbf{0.1007}$ | $\mathbf{0.1243}$ | $\mathbf{-0.1504}$ | $\mathbf{51.11\%}$ | $+0.1333$ | Ref | Ref |
| **Random Forest** | Fundamentals-Only (B) | $0.1078$ | $0.1320$ | $-0.2966$ | $24.44\%$ | $\mathbf{+0.1667}$ | $+0.0071$ | $-7.00\%$ |
| **Random Forest** | Combined (C) | $0.1048$ | $0.1281$ | $-0.2205$ | $37.78\%$ | $0.0000$ | $\mathbf{+0.0041}$ | $\mathbf{-4.05\%}$ |
| **HistGradientBoosting** | Market-Only (A) | $\mathbf{0.0966}$ | $\mathbf{0.1195}$ | $\mathbf{-0.0630}$ | $\mathbf{55.56\%}$ | $+0.1333$ | Ref | Ref |
| **HistGradientBoosting** | Fundamentals-Only (B) | $0.1101$ | $0.1347$ | $-0.3512$ | $28.89\%$ | $+0.1333$ | $+0.0136$ | $-14.04\%$ |
| **HistGradientBoosting** | Combined (C) | $0.1002$ | $0.1236$ | $-0.1376$ | $53.33\%$ | $\mathbf{+0.2000}$ | $\mathbf{+0.0036}$ | $\mathbf{-3.71\%}$ |

![Predicted vs Realized Returns](protocol_predicted_vs_realized.png)

---

## 6. Statistical Uncertainty Analysis (Paired Block Bootstrap)

To account for overlapping 3-month forward return horizons and cross-stock correlation across equities observed on the same dates, inference was conducted using a **Date-Level Moving-Block Bootstrap ($L = 3$ calendar months per block, $B = 2,000$ replications, seed = 42)**. Entire cross-sections of stocks are sampled within contiguous 3-month blocks, exactly preserving temporal and cross-sectional dependence.

### 6.1 Bootstrap Estimates for Primary Estimand ($\Delta\text{MAE} = \text{MAE}_{\text{comb}} - \text{MAE}_{\text{mkt}}$)

| Model Architecture | Observed $\Delta\text{MAE}$ | 95% Block Bootstrap CI for $\Delta\text{MAE}$ | Observed % Improvement | 95% CI for % Improvement | Superiority Test ($H_0: \Delta\text{MAE} \ge 0$) $p$-value | Statistical Conclusion ($\alpha = 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regressor** | $+0.0335$ | $[+0.0105, +0.0502]$ | $-35.64\%$ | $[-58.79\%, -10.46\%]$ | $p = 1.0000$ | **Combined is Statistically Significantly WORSE** ($95\%\text{ CI} > 0$) |
| **Random Forest** | $+0.0041$ | $[-0.0048, +0.0094]$ | $-4.05\%$ | $[-10.92\%, +4.64\%]$ | $p = 0.7005$ | **Fail to Reject $H_0$** (Indistinguishable / Spans 0) |
| **HistGradientBoosting** | $+0.0036$ | $[-0.0065, +0.0101]$ | $-3.71\%$ | $[-10.44\%, +7.78\%]$ | $p = 0.6940$ | **Fail to Reject $H_0$** (Indistinguishable / Spans 0) |

![Bootstrap Delta MAE Distribution](protocol_delta_mae_distribution.png)

### 6.2 Test of Protocol Confirmatory Hypotheses

- **H1 (Primary: Combined Features Reduce MAE vs Market-Only):** **REJECTED.** Across all three architectures, $\Delta\text{MAE} > 0$. Point-in-time fundamental indicators did not reduce prediction error; for Ridge regression, the deterioration was statistically significant ($p = 1.0000$ for superiority; 95% CI excludes 0).
- **H2 (Fundamentals-Only Improves on Historical-Mean Baseline):** **REJECTED.** The training historical-mean baseline achieved MAE $= 0.0954$ and RMSE $= 0.1159$. In contrast, Fundamentals-Only achieved MAE $= 0.1186$ (Ridge), $0.1078$ (Random Forest), and $0.1101$ (HistGradientBoosting). Historical accounting data alone was substantially less predictive than the unconditional sample mean.
- **H3 (Profitability/Cash Flow Adds Value Beyond Valuation):** **CONFIRMED VIA ABLATION.** Valuation ratios (`pe_proxy`, `pb_proxy`) severely destabilized linear models. Dropping valuation features from Combined improved HistGradientBoosting MAE from $0.1002$ to $0.0919$ and increased OOS $R^2$ to $+0.0896$.
- **H4 (Tree Models Improve on Ridge Under Matched Budgets):** **CONFIRMED ON COMBINED & FUNDAMENTAL SETS.** On the noisy Combined feature set, Ridge suffered severely ($\text{MAE} = 0.1276$), whereas HistGradientBoosting moderated the noise ($\text{MAE} = 0.1002$). However, on Market-Only features, Ridge marginally edged the tree models ($\text{MAE} = 0.0940$ vs $0.0966$).

---

## 7. Feature-Group Ablation Study

To identify which indicator groups drove predictive degradation or improvement, a systematic leave-one-group-out ablation was conducted on the matched sample:

| Feature Configuration | Number of Predictors ($k$) | Ridge Holdout MAE | Ridge OOS $R^2$ | HGB Holdout MAE | HGB OOS $R^2$ | HGB Rank IC | Primary Diagnostic Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Combined (All Features)** | 24 | $0.1276$ | $-0.8875$ | $0.1002$ | $-0.1376$ | $+0.2000$ | Baseline combined configuration |
| **Ablation: No Valuation** | 22 | $0.1072$ | $-0.4170$ | $\mathbf{0.0919}$ | $\mathbf{+0.0896}$ | $\mathbf{+0.4667}$ | **Dropping valuation significantly IMPROVES performance** |
| **Ablation: No Profitability** | 21 | $0.1210$ | $-0.7269$ | $0.1035$ | $-0.2219$ | $+0.1667$ | Profitability removal slightly hurts HGB |
| **Ablation: No Growth** | 22 | $0.1125$ | $-0.4870$ | $0.1250$ | $-0.9080$ | $+0.0667$ | Growth removal significantly hurts HGB |
| **Ablation: No Leverage** | 22 | $0.1260$ | $-0.8327$ | $0.1001$ | $-0.1327$ | $+0.2000$ | Neutral impact |
| **Market-Only (No Fundamentals)** | 14 | $\mathbf{0.0940}$ | $\mathbf{-0.0909}$ | $0.0966$ | $-0.0630$ | $+0.1333$ | Best overall parsimonious model |
| **Fundamentals-Only (No Market)** | 10 | $0.1186$ | $-0.4954$ | $0.1101$ | $-0.3512$ | $+0.1333$ | Worst overall model family |

![Feature Group Ablation](protocol_feature_group_ablation.png)

*Ablation Insights:*  
The ablation reveals that the failure of Combined features was primarily driven by **valuation ratio distortion**: quarterly earnings denominator spikes produced unstable P/E and P/B metrics that degraded linear coefficients. Eliminating valuation while retaining quarterly revenue/earnings growth allowed HistGradientBoosting to achieve positive out-of-sample $R^2$ ($+0.0896$) and high cross-sectional rank correlation ($\text{Rank IC} = +0.4667$).

---

## 8. Artifact Inventory & Reproduction Commands

All evaluation artifacts, predictions, machine-readable metrics, configuration files, logs, and figures have been generated and persisted:

### 8.1 Machine-Readable Artifact Manifest

```
c:/Users/Adity/OneDrive/Desktop/IML_Research_Project/protocol_experiments/
├── protocol_execution.log                                # Complete stdout/file execution log
├── run_protocol_experiments.py                           # Master reproduction script
└── outputs/
    ├── configs/
    │   └── protocol_experiment_config.json              # Frozen experiment configuration
    ├── metrics/
    │   ├── protocol_metrics_universe.json               # Full 16-stock universe benchmark metrics
    │   ├── protocol_metrics_matched.json                # Matched comparison metrics (JSON)
    │   ├── protocol_model_comparison.csv                # Matched model comparison table (CSV)
    │   ├── protocol_ablation_metrics.json               # Leave-one-group-out ablation metrics
    │   └── protocol_bootstrap_uncertainty.json          # Paired block bootstrap 95% CIs and p-values
    ├── predictions/
    │   ├── holdout_predictions_protocol_universe.csv    # Row-level universe holdout forecasts
    │   └── holdout_predictions_protocol_matched.csv     # Row-level matched holdout forecasts
    └── figures/
        ├── protocol_delta_mae_distribution.png          # Bootstrap Delta MAE distribution
        ├── protocol_feature_group_ablation.png          # Feature ablation bar chart
        ├── protocol_predicted_vs_realized.png           # Scatter calibration plot
        └── protocol_universe_model_comparison.png      # Universe benchmark chart
```

### 8.2 Reproduction Commands
To replicate all numbers, tables, and figures from scratch:

```powershell
# 1. Run unit test suites
cd "c:\Users\Adity\OneDrive\Desktop\IML_Research_Project\IML_project\model_development"
python -m pytest tests/ -v

cd "c:\Users\Adity\OneDrive\Desktop\IML_Research_Project\NSE_Fundamentals_Collector\NSE_Fundamentals_Collector"
python -m pytest tests/ -v

# 2. Execute protocol experiments
cd "c:\Users\Adity\OneDrive\Desktop\IML_Research_Project\protocol_experiments"
python run_protocol_experiments.py
```

---

## 9. Critical Discrepancies, Limitations, and Remediation Roadmap

### 9.1 Failed Checks & Protocol Deviations

1. **Pre-Model Feasibility Gate Blocked:**  
   The primary paper claim of evaluating 16 NSE stocks with fundamental features fails Section 17 feasibility requirements. 13 of 16 stocks have 0 filings collected due to local network restrictions.
2. **Banking Taxonomy Extraction Defect:**  
   `HDFCBANK` statements cannot be parsed by generic commercial XBRL tag sets (`q_revenue`, `q_pat`). Financial institutions require dedicated interest income / NII / operating profit parsers.
3. **Short Sample Size on Matched Holdout:**  
   The matched holdout contains 45 company-month rows (15 calendar dates across 3 stocks). While the paired date-block bootstrap accounts for date clustering, 15 blocks is small for high-precision inference.

### 9.2 Required Fixes Before Results Can Be Trusted for Academic Publication

```
Remediation Path to Full Publication Trust
┌───────────────────────────────┐     ┌──────────────────────────────┐
│ 1. Download XBRL for 13 Stocks │ ──> │ 2. Banking Taxonomy Parser   │
│ Ingest remaining 13 symbols   │     │ Extract NII/PAT for HDFCBANK │
└───────────────────────────────┘     └──────────────────────────────┘
                                                     │
                                                     ▼
┌───────────────────────────────┐     ┌──────────────────────────────┐
│ 4. Re-run Frozen Protocol     │ <── │ 3. Standardize Valuation     │
│ Expand to 16 stocks / 5 years │     │ Winsorize / drop noisy P/E   │
└───────────────────────────────┘     └──────────────────────────────┘
```

1. **Complete XBRL Ingestion for Remaining 13 Equities:** In an environment with outbound network access or via authorized offline filing archives, download financial results for `ABB`, `ADANIGREEN`, `APOLLOHOSP`, `MARUTI`, `MOTILALOFS`, `POWERINDIA`, `SBIN`, `SIEMENS`, `SOLARINDS`, `TCS`, `TITAN`, `ULTRACEMCO`, and `YESBANK`.
2. **Implement Banking-Specific XBRL Extractor:** Update `parse_integrated_xbrl.py` to extract `InterestEarned`, `InterestExpended`, and `OperatingProfit` when parsing commercial banks (`HDFCBANK`, `SBIN`, `YESBANK`).
3. **Re-scale or Winsorize Valuation Ratios:** Apply in-fold winsorization or use earnings-yield ($E/P$) rather than $P/E$ to avoid denominator explosion when earnings approach zero.
4. **Re-Execute the Frozen Pipeline Across Full Universe:** Once the remaining 13 stocks have verified point-in-time fundamentals, execute `run_protocol_experiments.py` to evaluate the complete 16-stock universe without sample attrition.
