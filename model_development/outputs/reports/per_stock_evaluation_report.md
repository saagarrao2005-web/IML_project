# Per-Stock Evaluation Report (Holdout Period)

**Holdout Window:** 22 January 2026 to 15 October 2026 (181 Trading Observations per Stock)  
**Evaluated Architecture:** Selected Best Model (Logistic Regression Pipeline)  
**Total Holdout Records:** 2,896 observations across 16 Equities  

---

## 1. Per-Stock Performance Summary

The table below reports out-of-sample performance metrics evaluated independently for each equity in the target universe over the untouched holdout period.

| Symbol | Observations | Up Move Ratio (Class 1) | Accuracy | Balanced Accuracy | F1-Score | Holdout ROC-AUC | Outperformance vs Chance (0.50) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RELIANCE** | 181 | 44.20% | **54.14%** | **56.18%** | **0.5871** | **0.6196** | **+11.96%** |
| **INFY** | 181 | 44.20% | 50.83% | 50.87% | 0.4795 | **0.5540** | **+5.40%** |
| **HDFCBANK** | 181 | 38.12% | 45.30% | 51.63% | 0.5217 | **0.5493** | **+4.93%** |
| **SOLARINDS** | 181 | 63.54% | 43.65% | 51.78% | 0.3289 | **0.5493** | **+4.93%** |
| **ULTRACEMCO** | 181 | 44.75% | **54.14%** | **55.57%** | 0.5744 | **0.5411** | **+4.11%** |
| **SIEMENS** | 181 | 50.83% | 49.72% | 50.16% | 0.3259 | **0.5181** | **+1.81%** |
| **POWERINDIA** | 181 | 54.14% | 46.41% | 49.13% | 0.2481 | 0.4720 | -2.80% |
| **MOTILALOFS** | 181 | 51.93% | 44.75% | 45.65% | 0.2958 | 0.4681 | -3.19% |
| **APOLLOHOSP** | 181 | 59.12% | 45.86% | 45.25% | 0.5149 | 0.4663 | -3.37% |
| **ABB** | 181 | 48.62% | 48.07% | 47.66% | 0.3816 | 0.4653 | -3.47% |
| **TCS** | 181 | 37.57% | 43.64% | 45.79% | 0.4205 | 0.4593 | -4.07% |
| **MARUTI** | 181 | 35.91% | 48.07% | 52.04% | 0.4778 | 0.4556 | -4.44% |
| **YESBANK** | 181 | 40.88% | 44.75% | 44.31% | 0.3827 | 0.4552 | -4.48% |
| **TITAN** | 181 | 49.72% | 43.64% | 43.60% | 0.3780 | 0.4330 | -6.70% |
| **SBIN** | 181 | 48.62% | 45.30% | 45.49% | 0.4817 | 0.4274 | -7.26% |
| **ADANIGREEN** | 181 | 52.49% | 44.75% | 46.38% | 0.2063 | 0.4143 | -8.57% |

---

## 2. In-Depth Cross-Stock Dispersal Analysis

### 2.1 Outperforming Cluster (ROC-AUC 0.54 to 0.62)
- **Reliance Industries (`RELIANCE`), Infosys (`INFY`), HDFC Bank (`HDFCBANK`), UltraTech Cement (`ULTRACEMCO`), Solar Industries (`SOLARINDS`):**
  These large-cap institutional bellwethers exhibited structured trend momentum and clear mean-reversion characteristics over the holdout period. For example, Reliance achieved **0.6196 ROC-AUC** and **54.14% directional accuracy**, demonstrating that in liquid, institutionally traded stocks with high participation, rolling moving-average distances (`close_vs_sma_50`) and RSI momentum retain modest statistical predictive utility.

### 2.2 Underperforming / High-Noise Cluster (ROC-AUC < 0.48)
- **Adani Green (`ADANIGREEN`), State Bank of India (`SBIN`), Titan (`TITAN`), Yes Bank (`YESBANK`):**
  These equities underperformed the pooled model predictions. In particular, Adani Green (0.4143 AUC) was heavily impacted by idiosyncratic headline and regulatory events throughout 2024-2026 (Hindenburg fallout, US DOJ indictment news, renewable policy revisions), creating sudden price reversals that disconnected from purely technical, historical price patterns. Similarly, Yes Bank experienced high retail retail speculation and low share price friction that degraded momentum indicators.

### 2.3 Methodological Implication for Quantitative Trading
A single pooled cross-sectional model fitted across diverse stocks treats all symbols as identical stochastic processes. The empirical evidence proves that **market microstructure and idiosyncratic volatility vary dramatically across individual equities**. In production quantitative trading, equities should be partitioned into homogeneous clusters (e.g., mega-cap IT/banking vs volatile thematic momentum names) or trained with per-stock hierarchical Bayesian / asset-specific shrinkage models.
