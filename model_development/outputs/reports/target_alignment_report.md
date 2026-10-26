# Target Construction and Alignment Audit Report

**Prediction Horizon:** 5 Trading Sessions ($t$ to $t+5$)  
**Target Label:** Binary Directional Indicator (`target_up_5d`)  
**Continuous Regression Target:** Forward 5-Session Return (`future_return_5d`)  

---

## 1. Mathematical Formulation

### 1.1 Binary Classification Target
For each equity symbol $s$ at trading session $t$:
$$\text{future\_adj\_close\_5d}_{s, t} = \text{adj\_close}_{s, t+5}$$
$$\text{future\_return\_5d}_{s, t} = \frac{\text{adj\_close}_{s, t+5}}{\text{adj\_close}_{s, t}} - 1.0$$

The binary target label is defined as:
$$\text{target\_up\_5d}_{s, t} = \begin{cases} 
1.0 & \text{if } \text{future\_adj\_close\_5d}_{s, t} > \text{adj\_close}_{s, t} \\
0.0 & \text{if } \text{future\_adj\_close\_5d}_{s, t} \le \text{adj\_close}_{s, t} \\
\text{NaN} & \text{if } t > T_s - 5
\end{cases}$$

Where $T_s$ is the final recorded observation for stock $s$.

### 1.2 Tie-Breaking Convention
In the event that the closing price 5 trading observations ahead exactly equals the current closing price ($\text{future\_return\_5d} == 0$), the target is assigned to class `0.0`. Across the 19,840-observation dataset, exactly **39 tie instances** were identified, representing **0.197%** of observations. All 39 instances are mapped to `0.0` consistently.

---

## 2. Per-Symbol Alignment and Boundary Isolation

A critical hazard in pooled multi-asset time series modeling is cross-symbol contamination during forward lag shifting. If observations are sorted only by date or shifted without grouping by ticker, observations from stock $A$ can bleed into the forward target of stock $B$.

### Verification Protocol
1. **Grouping Assertion:** All shifts are executed strictly within `groupby('Symbol', sort=False)`.
2. **Boundary Testing:** An automated test verifies that for every row $i$, the symbol at row $i$ matches the symbol at row $i+5$, and that the forward close matches `adj_close[i+5]`.
3. **Truncation Verification:** Exactly $16 \times 5 = 80$ rows in the analytical dataset have an unavailable target (`target_up_5d.isna() == True`). These correspond exclusively to the final 5 calendar dates for each of the 16 stocks.

**Result: 100% Verified. Zero cross-symbol leakage.**

---

## 3. Impact of Corporate Action Adjustments on Target Integrity

In the unadjusted legacy dataset, major corporate actions severely corrupted target labels:
- **MOTILALOFS (June 2024, 3:1 Bonus):** In the 5 sessions preceding the ex-date, raw prices compared pre-bonus (~INR 2,400) to post-bonus (~INR 650), yielding artificial forward returns of $-72\%$ and assigning target `0` to all 5 sessions. On an adjusted basis, the stock gained $+8.5\%$ to $+28.0\%$ over those spans, correctly assigning target `1`.
- **RELIANCE (October 2024, 1:1 Bonus):** 5 pre-event sessions previously suffered from an artificial $-50\%$ return and label `0`. On an adjusted basis, 2 of those sessions were positive, flipping to label `1`.
- **SIEMENS (April 2025, Energy Demerger):** 5 pre-demerger sessions previously showed a fake $-45\%$ drop and label `0`. On an adjusted basis, Siemens gained $+3.4\%$ to $+15.9\%$, correctly receiving label `1`.
- **HDFCBANK (August 2025, 1:1 Bonus):** 5 pre-event sessions previously showed a $-51\%$ return. On an adjusted basis, the returns are normalized to true economic price movements ($-2.2\%$ to $-4.4\%$).

---

## 4. Class Distribution Summary

| Metric | Full Analytical Dataset | Development Set (85%) | Holdout Set (15%) |
| :--- | :--- | :--- | :--- |
| Total Labeled Observations | 19,760 | 16,864 | 2,896 |
| Class 1 (Upward Move) | 10,328 (52.27%) | 8,944 (53.04%) | 1,384 (47.79%) |
| Class 0 (Downward / Tie) | 9,432 (47.73%) | 7,920 (46.96%) | 1,512 (52.21%) |
| Class Balance Ratio (1 / 0) | 1.095 | 1.129 | 0.915 |

*Note: The mild class imbalance reflects typical macroeconomic regimes (a moderate bull market during the 2021-2025 development window, transitioning into consolidation in the 2026 holdout window).*
