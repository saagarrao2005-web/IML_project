# Data Quality and Target Leakage Audit Report

**Project:** NSE Stock Price Prediction using Machine Learning  
**Universe:** 16 NSE Equities (ABB, ADANIGREEN, APOLLOHOSP, HDFCBANK, INFY, MARUTI, MOTILALOFS, POWERINDIA, RELIANCE, SBIN, SIEMENS, SOLARINDS, TCS, TITAN, ULTRACEMCO, YESBANK)  
**Sample Period:** 25 October 2021 to 23 October 2026 (1,240 trading observations per symbol)  
**Total Records:** 19,840 observations  

---

## 1. Executive Summary

This audit establishes the integrity, consistency, and point-in-time validity of the daily equity market dataset used in the NSE Stock Price Prediction system. Rigorous tests were conducted across key data dimensions:
- Series selection and multi-series duplication resolution.
- Date parsing and temporal monotonicity.
- Symbol-Date primary key uniqueness.
- Internal OHLC price consistency and bounds.
- Traded volume, trade count, and delivery quantity sanity.
- Look-ahead bias and feature leakage prevention.

All automated assertions passed with zero exceptions.

---

## 2. Ingestion and Series Filtering Audit

### 2.1 Multi-Series Root Cause Analysis
The original raw exchange files contained multiple NSE trading series on the same trading dates, notably:
- `EQ`: Standard rolling settlement equity shares.
- `BL`: Block trade window transactions.
- `W3`: Special institutional or settlement series.

Earlier baseline attempts suffered from duplicate `(Symbol, Date)` keys because rows across multiple series shared the same date.

### 2.2 Resolution and Verification
A dedicated equity series extraction pipeline was executed:
1. Filtered strictly for `Series == 'EQ'`.
2. Verified that each of the 16 equity files contains exactly 1,240 observations.
3. Total filtered records across the universe: **19,840 rows**.
4. **Duplicate Symbol-Date keys: 0** (verified across all 19,840 records).
5. **Invalid/Unparseable dates: 0** (all dates normalize to valid exchange business dates).

---

## 3. Price Validity and OHLC Sanity Audit

Every price record was converted from Indian number formatting (stripping commas and Rupee currency symbols) and subjected to strict mechanical boundary checks:

1. **Non-Negativity:**
   - Minimum closing price: INR 12.10 (YESBANK).
   - Closing prices $\le 0$: **0 records**.
2. **High-Low Ordering:**
   - $\text{High} \ge \text{Low}$ across all records: **100% compliant (0 violations)**.
3. **Open-Close Bounding:**
   - $\text{High} \ge \max(\text{Open}, \text{Close})$: **100% compliant (0 violations)**.
   - $\text{Low} \le \min(\text{Open}, \text{Close})$: **100% compliant (0 violations)**.
4. **Volume and Liquidity Bounding:**
   - Traded volume $\le 0$: **0 records** (min volume = 1,070 shares).
   - Number of trades $\le 0$: **0 records** (min trades = 322).
   - Deliverable quantity $\le 0$: **0 records** (min deliverable = 743 shares).

---

## 4. Look-Ahead Bias and Feature Leakage Safeguards

To ensure that reported out-of-sample performance is entirely defensible and free from survivorship or future-information leaks, the following architectural controls are enforced:

### 4.1 Chronological Partitioning with Embargo Buffer
- All cross-validation and test partitions are partitioned strictly along distinct trading dates.
- Observations from the same calendar date are never split between training and evaluation folds.
- **5-Session Purge/Embargo Window:** Because the prediction target spans 5 forward trading sessions ($t$ to $t+5$), evaluating on session $t+1$ immediately after training on session $t$ would leak overlapping return information. The pipeline enforces a minimum 5-trading-day purge between the end of any training window and the start of any validation/holdout window.

### 4.2 In-Fold Preprocessing
- No global imputers, scalers, or feature selectors are fitted across the entire dataset.
- In both walk-forward validation and final holdout testing, `SimpleImputer` and `StandardScaler` are fitted exclusively on the training fold observations and applied out-of-sample.

### 4.3 Feature Availability Timestamp
- Every predictive feature is computed using information known at or before market close on session $t$.
- Features never incorporate future prices, forward returns, target labels, or post-close exchange announcements.

---

## 5. Audit Results Matrix

| Audit Check | Required Standard | Observed Result | Status |
| :--- | :--- | :--- | :--- |
| Unique Symbols | Exactly 16 expected symbols | 16 symbols confirmed | **PASS** |
| Symbol-Date Uniqueness | 0 duplicate keys | 0 duplicates found | **PASS** |
| Date Monotonicity | Chronologically increasing per symbol | 100% monotonic | **PASS** |
| OHLC Consistency | High $\ge$ Low, Open, Close; Low $\le$ Open, Close | 0 violations across 19,840 rows | **PASS** |
| Volume Sanity | Volume, Trades, Deliverable $> 0$ | All positive | **PASS** |
| Purge Buffer | $\ge 5$ trading sessions between train/test | 5-session embargo enforced | **PASS** |
| Target Shift | Strictly shifts within symbol | 0 cross-symbol bleed | **PASS** |
| Missing Target Handling | Last 5 sessions per symbol NaN | Exactly 80 rows NaN (16 $\times$ 5) | **PASS** |
