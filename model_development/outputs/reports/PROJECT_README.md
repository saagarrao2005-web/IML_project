# Submission Guide & Project Documentation: NSE Stock Price Prediction ML Project

**Course:** B.Tech Semester Project — Introduction to Machine Learning (IML)  
**Academic Year:** 2026  
**Repository Path:** `D:\B.Tech\Sem3\IML\Project\Stock_Prediction\model_development`  

---

## 1. Project Purpose

This project implements an end-to-end, scientifically defensible machine learning pipeline to predict the 5-trading-session directional price movement of 16 major National Stock Exchange (NSE) equities over a 5-year historical horizon (October 2021 to October 2026).

The system addresses the primary failure modes of financial machine learning:
1. **Multi-Series Data Corruption:** Isolates pure equity (`EQ`) series data, eliminating duplicate Symbol-Date records.
2. **Corporate Action Discontinuities:** Primary-source verified and retrospectively adjusted mechanical price actions (Motilal Oswal 3:1 bonus, Reliance 1:1 bonus, Siemens energy demerger, HDFC Bank 1:1 bonus) without leaking future information into historical features.
3. **Look-Ahead Bias & Target Leakage:** Enforces expanding-window chronological cross-validation with an exact 5-trading-day purge/embargo buffer and strictly in-fold preprocessing.
4. **Honest Evaluation:** Models are evaluated on an untouched chronological holdout set (2,896 observations) using Date-Level Moving-Block Bootstrap confidence intervals.

---

## 2. Requirements & Installation

The project uses standard, lightweight Python machine learning packages:
- Python 3.10 to 3.12 (Tested on Python 3.12.10)
- `numpy>=1.26.0`
- `pandas>=2.0.0`
- `scipy>=1.12.0`
- `scikit-learn>=1.4.0`
- `xgboost>=2.0.0`
- `joblib>=1.3.0`
- `matplotlib>=3.8.0`
- `seaborn>=0.13.0`
- `pytest>=8.0.0`

### Installation Command
From the project directory:
```bash
cd "D:\B.Tech\Sem3\IML\Project\Stock_Prediction\model_development"
pip install -r requirements.txt
```

---

## 3. Directory Structure

```text
model_development/
├── config.py                 # Central configurations, paths, registers, and parameters
├── data_pipeline.py          # Data loading, quality audits, adjustments, features & target
├── train_evaluate.py         # Chronological walk-forward validation and holdout evaluation
├── predict.py                # Command-line and programmatic prediction interface
├── run_project.py            # Master runner orchestrating the complete pipeline
├── requirements.txt          # Dependency specification file
├── legacy_scripts_backup/    # Archived exploratory scripts
├── tests/
│   └── test_pipeline.py      # Automated pytest suite (10 unit/integration tests)
└── outputs/
    ├── datasets/
    │   └── analytical_dataset.csv   # Clean 19,840-row analytical dataset
    ├── models/
    │   └── best_model.joblib        # Serialized fitted pipeline
    ├── predictions/
    │   └── holdout_predictions.csv  # Holdout predictions with probabilities
    ├── figures/                     # ROC curves, confusion matrix, calibration plots
    └── reports/                     # Audit, evaluation, and final project documentation
```

---

## 4. How to Run the Pipeline

### One-Command Full Pipeline Execution
To execute the complete pipeline (data ingestion, data quality audit, corporate action adjustment, feature engineering, target construction, walk-forward validation, holdout evaluation, figure generation, and automated test suite):
```bash
python run_project.py
```
*Average execution time: ~45 seconds on standard hardware.*

---

## 5. How to Run Automated Tests

To run the automated unit and integration tests:
```bash
pytest tests/ -v
```
**Test Suite Coverage (10 Tests):**
- Ingestion and date parsing monotonicity.
- Symbol-Date uniqueness (zero duplicate keys).
- Equity series enforcement.
- OHLC consistency and price bounds.
- Corporate action adjustment direction and continuity.
- Per-symbol shift isolation and target alignment.
- Absence of future leakage in feature columns.
- Chronological purge buffer separation.
- Prediction interface schema and output bounds.

---

## 6. How to Generate Predictions

Predictions are generated via `predict.py` using completed daily market data through observation $t$:

### Predict a Single Symbol
```bash
python predict.py --symbol RELIANCE
```
*Sample Output:*
```text
=================================================================
NSE 5-DAY STOCK DIRECTIONAL FORECAST INTERFACE
=================================================================

Symbol:               RELIANCE
As of Market Close:   2026-10-23
Current Close:        INR 1,190.00
Prediction Horizon:   5 trading sessions ahead
Forecast Direction:   UP (Bullish)
Probability Up (5d):  52.5%
Probability Down (5d):47.5%
Confidence Level:     Low / Near-Neutral (Close to 50/50 Chance)

MODEL ESTIMATE ONLY: This directional forecast is an empirical statistical estimate derived from technical market indicators. It does NOT predict an exact target price and does NOT guarantee market performance or profitability.
=================================================================
```

### Predict Across the Entire 16-Stock Universe
```bash
python predict.py --all
```

### Launch Interactive Web Dashboard
To launch the Streamlit web application:
```bash
streamlit run app.py
```
Or from the project root:
```bash
streamlit run model_development/app.py
```
Visit `http://localhost:8501` to view the full interactive dashboard, individual stock price charts, and performance reports.


---

## 7. Understanding the Prediction Output

- **`Forecast Direction`:** `UP (Bullish)` if the predicted probability of an upward move is $\ge 50\%$; `DOWN (Bearish)` if $< 50\%$.
- **`Probability Up (5d)`:** The model's estimated posterior probability that the closing price 5 trading sessions ahead will be strictly greater than the current session's closing price.
- **`Confidence Level`:**
  - *Low / Near-Neutral:* Probability is within $\pm 3\%$ of $50\%$ (close to a coin flip).
  - *Moderate Statistical Skew:* Probability is between $53\% - 60\%$ or $40\% - 47\%$.
  - *Distinct Directional Skew:* Probability is $> 60\%$ or $< 40\%$.

---

## 8. Known Limitations & Academic Disclaimers

1. **Market Efficiency:** In liquid equities, daily market data exhibits low signal-to-noise ratio. Real-world out-of-sample directional ROC-AUC hovers around $0.51 - 0.52$.
2. **Not Financial Advice:** Model outputs are empirical statistical estimates for academic research. They must not be used for live capital allocation or financial trading.
3. **Data Freshness:** The model requires completed daily Bhavcopy market data. It does not interpolate missing intraday feeds.
