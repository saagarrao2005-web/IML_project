# NSE Stock Price Prediction using Machine Learning

A quantitative machine learning project designed to predict 5-trading-session directional price movements for major National Stock Exchange (NSE) equities using chronological validation, corporate action reconciliation, and leakage-safe feature pipelines.

---

## 1. Project Overview

- **Target Universe:** 16 NSE Equities (`ABB`, `ADANIGREEN`, `APOLLOHOSP`, `HDFCBANK`, `INFY`, `MARUTI`, `MOTILALOFS`, `POWERINDIA`, `RELIANCE`, `SBIN`, `SIEMENS`, `SOLARINDS`, `TCS`, `TITAN`, `ULTRACEMCO`, `YESBANK`).
- **Sample Period:** 25 October 2021 to 23 October 2026 (5 Full Years, 1,240 trading sessions per stock, 19,840 total records).
- **Prediction Objective:** Predict whether the closing price 5 trading sessions ahead ($t+5$) is higher than the closing price at session $t$ ($1 = \text{Up}, 0 = \text{Down/Tie}$).
- **Validation Protocol:** Expanding-window walk-forward cross-validation (3 chronological folds) with 5-session purging/embargo buffer, followed by evaluation on an untouched final 15% chronological holdout (2,896 observations).

---

## 2. Directory Structure

```text
model_development/
├── config.py                 # Central configuration, paths, corporate actions, and hyperparams
├── data_pipeline.py          # Ingestion, data auditing, corporate action adjustment, features, target
├── train_evaluate.py         # Chronological walk-forward validation, holdout evaluation, plotting
├── predict.py                # Command-line and programmatic prediction interface
├── run_project.py            # Master pipeline runner executing end-to-end workflow
├── requirements.txt          # Python dependencies
├── legacy_scripts_backup/    # Timestamped archive of legacy numbered exploratory scripts
├── tests/                    # Automated pytest test suite
│   └── test_pipeline.py      # Comprehensive unit and integration tests (10 passing tests)
├── outputs/                  # Generated artifacts
│   ├── datasets/             # Clean analytical dataset (analytical_dataset.csv)
│   ├── models/               # Serialized trained model pipeline (best_model.joblib)
│   ├── predictions/          # Out-of-sample holdout predictions (holdout_predictions.csv)
│   ├── figures/              # High-resolution diagnostic charts (ROC, PR, calibration, feature importance)
│   └── reports/              # Comprehensive audits, model comparisons, and final report
└── README.md
```

---

## 3. Quick Start & Execution Instructions

### 3.1 Prerequisites
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 3.2 Run the Complete Pipeline
Execute the full end-to-end pipeline (data preparation, auditing, modeling, evaluation, plotting, and testing):
```bash
python run_project.py
```

### 3.3 Run Automated Tests
Execute the unit and integration test suite:
```bash
pytest tests/ -v
```

### 3.4 Generate Stock Predictions
Generate a 5-day directional forecast for a specific stock:
```bash
python predict.py --symbol RELIANCE
```

Generate forecasts across all 16 universe stocks:
```bash
python predict.py --all
```

### 3.5 Launch Interactive Web Dashboard
Launch the clean, professional Streamlit web dashboard:
```bash
streamlit run app.py
```
Or with Python:
```bash
python -m streamlit run app.py
```
Then open your browser at `http://localhost:8501`.
Features include:
- **Market Dashboard:** 16-stock directional forecasts, probability skews, and filtering.
- **Stock Prediction:** Individual forecasts, price history, moving average overlays, and recent sessions.
- **Model Performance:** Holdout comparison tables, walk-forward validation results, and diagnostic figures.
- **About Project:** Mathematical problem formulation, feature engineering catalog, and disclaimers.


---

## 4. Key Quantitative Results

- **Data Integrity:** 19,840 rows, 0 duplicate keys, 0 invalid dates, 0 OHLC violations.
- **Corporate Actions Reconciled:** 4 confirmed mechanical price actions adjusted retrospectively (Motilal Oswal 3:1 bonus, Reliance 1:1 bonus, Siemens energy demerger, HDFC Bank 1:1 bonus).
- **Walk-Forward Cross-Validation:** Logistic Regression achieved highest mean ROC-AUC of **$0.5159 \pm 0.0130$** (Balanced Acc: $0.5091$).
- **Untouched Holdout Evaluation:** HistGradientBoosting achieved **$0.5187$ ROC-AUC**, **$51.80\%$ Accuracy**, and **$52.23\%$ Balanced Accuracy**, beating the chance baseline.
- **Individual Stock Dispersion:** Reliance Industries achieved **$0.6196$ Holdout AUC** ($54.1\%$ accuracy) and Infosys achieved **$0.5540$ AUC**, whereas volatile/headline-driven stocks (Adani Green, 0.4143 AUC) degraded.
- **Defensibility:** In strict accordance with the Efficient Market Hypothesis, results show that short-term daily equity price moves possess low signal-to-noise ratio. Reported metrics are completely free of look-ahead bias and forward leakage.

---

## 5. Submission & Audit Documentation

For full academic research documentation and verification reports:
- **Comprehensive Project Report (15 Sections):** [`outputs/reports/FINAL_PROJECT_REPORT.md`](outputs/reports/FINAL_PROJECT_REPORT.md)
- **Detailed Project README:** [`outputs/reports/PROJECT_README.md`](outputs/reports/PROJECT_README.md)
- **Final Submission Checklist:** [`outputs/reports/FINAL_SUBMISSION_CHECKLIST.md`](outputs/reports/FINAL_SUBMISSION_CHECKLIST.md)
- **Independent Audit & Sign-Off Report:** [`outputs/reports/FINAL_INDEPENDENT_AUDIT.md`](outputs/reports/FINAL_INDEPENDENT_AUDIT.md)

