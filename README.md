# NSE Stock Price Prediction using Machine Learning

**B.Tech 3rd Semester — Introduction to Machine Learning (IML) Project**

This repository contains a machine learning system for predicting 5-trading-session directional price movements across 16 major National Stock Exchange of India (NSE) equities over a 5-year sample period (October 2021 to October 2026).

---

## Repository Structure

```text
Stock_Prediction/
├── app.py                    # Root Streamlit entry point (forwards to model_development/app.py)
├── requirements.txt          # Project dependencies
├── cleaned_Dataset/          # Cleaned raw NSE daily CSV files (16 stocks)
├── Datasets/                 # Original raw NSE Bhavcopy CSVs
├── model_development/        # Core ML system and quantitative pipeline
│   ├── app.py                # Streamlit dashboard (4 pages)
│   ├── config.py             # Paths, constants, corporate action registers
│   ├── data_pipeline.py      # Ingestion, quality audits, adjustments, features & target
│   ├── train_evaluate.py     # Chronological walk-forward validation and holdout evaluation
│   ├── predict.py            # CLI & programmatic prediction interface
│   ├── run_project.py        # One-command full pipeline execution
│   ├── requirements.txt      # Project dependencies
│   ├── tests/                # 10 automated test cases (pytest)
│   ├── outputs/              # Generated datasets, models, predictions, figures, and reports
│   └── README.md             # Detailed technical documentation
└── README.md                 # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard

From the **project root**:
```bash
streamlit run app.py
```

Or from the `model_development` directory:
```bash
cd model_development
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

The dashboard has four pages:
- **Market Dashboard** — Universe-wide predictions with filters and CSV export
- **Stock Prediction** — Individual stock forecasts, price charts, and technical indicators
- **Model Performance** — Holdout comparison, walk-forward validation, diagnostic figures
- **About Project** — Methodology, feature engineering, limitations, and disclaimers

### 3. Run Predictions from Command Line

```bash
cd model_development
python predict.py --symbol RELIANCE
python predict.py --all
```

### 4. Run the Full Pipeline (Data + Training + Evaluation)

```bash
cd model_development
python run_project.py
```

### 5. Run Automated Tests

```bash
cd model_development
python -m pytest tests/ -v
```

---

## Key Results

- **Dataset:** 19,840 daily records across 16 NSE equities (Oct 2021 – Oct 2026)
- **Corporate Actions:** 4 confirmed events back-adjusted (MOTILALOFS, RELIANCE, SIEMENS, HDFCBANK)
- **Best Holdout ROC-AUC:** 0.5187 (Hist Gradient Boosting)
- **Deployed Model:** Logistic Regression (selected for best walk-forward performance, ROC-AUC: 0.5159 ± 0.0130)
- **Conclusion:** Consistent with the Efficient Market Hypothesis, short-term directional predictability from daily technical features alone is near chance level

---

## Documentation

- [Detailed Technical README](model_development/README.md)
- [Final Project Report](model_development/outputs/reports/FINAL_PROJECT_REPORT.md)
- [Final Submission Checklist](model_development/outputs/reports/FINAL_SUBMISSION_CHECKLIST.md)
- [Independent Audit Report](model_development/outputs/reports/FINAL_INDEPENDENT_AUDIT.md)
