# Feature Definitions and Availability Report

This document records the exact mathematical formula, warmup window, missing-value behavior, and information availability timestamp for every predictive feature used in the NSE Stock Price Prediction ML system.

## 1. Information Availability Guarantee
All features are constructed strictly using market data published on or before the close of trading observation $t$. No future prices, targets, or overlapping returns are utilized in feature creation. Preprocessing (imputation and scaling) is fitted exclusively on training folds during cross-validation and evaluation to prevent look-ahead contamination.

## 2. Feature Catalog

| Feature Name | Category | Mathematical Definition | Warmup Period | Missing Value Handling |
| :--- | :--- | :--- | :--- | :--- |
| `return_1d` | Momentum | $P_{t} / P_{t-1} - 1$ | 1 session | Median imputed in training fold |
| `return_2d` | Momentum | $P_{t} / P_{t-2} - 1$ | 2 sessions | Median imputed in training fold |
| `return_5d` | Momentum | $P_{t} / P_{t-5} - 1$ | 5 sessions | Median imputed in training fold |
| `return_10d` | Momentum | $P_{t} / P_{t-10} - 1$ | 10 sessions | Median imputed in training fold |
| `return_20d` | Momentum | $P_{t} / P_{t-20} - 1$ | 20 sessions | Median imputed in training fold |
| `close_vs_sma_5` | Trend Ratio | $P_{t} / 	ext{SMA}_5(P) - 1$ | 5 sessions | Median imputed in training fold |
| `close_vs_sma_10` | Trend Ratio | $P_{t} / 	ext{SMA}_{10}(P) - 1$ | 10 sessions | Median imputed in training fold |
| `close_vs_sma_20` | Trend Ratio | $P_{t} / 	ext{SMA}_{20}(P) - 1$ | 20 sessions | Median imputed in training fold |
| `close_vs_sma_50` | Trend Ratio | $P_{t} / 	ext{SMA}_{50}(P) - 1$ | 50 sessions | Median imputed in training fold |
| `volatility_10d` | Risk | $	ext{Std}(	ext{return}_{1d}, 10)$ | 10 sessions | Median imputed in training fold |
| `volatility_20d` | Risk | $	ext{Std}(	ext{return}_{1d}, 20)$ | 20 sessions | Median imputed in training fold |
| `daily_range_pct` | Volatility | $(H_t - L_t) / P_t$ | 0 sessions | Fully available |
| `open_close_return` | Intraday | $P_t / O_t - 1$ | 0 sessions | Fully available |
| `volume_change_1d` | Liquidity | $V_t / V_{t-1} - 1$ | 1 session | Median imputed in training fold |
| `volume_vs_avg20` | Liquidity | $V_t / 	ext{SMA}_{20}(V) - 1$ | 20 sessions | Median imputed in training fold |
| `trades_vs_avg20` | Liquidity | $N_t / 	ext{SMA}_{20}(N) - 1$ | 20 sessions | Median imputed in training fold |
| `delivery_pct` | Quality | $	ext{Deliverable Qty} / 	ext{Traded Qty} 	imes 100$ | 0 sessions | Fully available |
| `rsi_14` | Oscillator | $100 - (100 / (1 + 	ext{RS}_{14}))$ | 14 sessions | Neutral 50.0 fallback |
| `day_of_week` | Calendar | Integer 0 (Mon) to 4 (Fri) | 0 sessions | Fully available |
| `month` | Calendar | Integer 1 (Jan) to 12 (Dec) | 0 sessions | Fully available |

*Note: All price ($P, H, L, O$) and volume ($V$) series are corporate-action adjusted backwards to prevent artificial jumps from stock splits, bonus issues, and demergers.*
