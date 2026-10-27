# NSE Stock Prediction — Quantitative Reproducibility Guide

This guide details the exact sequence of commands required to reproduce the data validation, automated testing, training pipeline execution, holdout evaluation, and Streamlit dashboard interface.

---

## 1. Environment & Prerequisites

- **Operating System:** Windows, Linux, or macOS.
- **Python Version:** Python 3.10 to 3.12 (Tested on Python 3.12.10).
- **Core Dependencies:** `numpy>=1.26.0`, `pandas>=2.0.0`, `scikit-learn>=1.4.0`, `xgboost>=2.0.0`, `scipy>=1.12.0`, `joblib>=1.3.0`, `pytest>=8.0.0`, `streamlit>=1.35.0`.

Install all dependencies from the root directory:
```bash
pip install -r requirements.txt
```

---

## 2. Step-by-Step Execution Sequence

### Step 1: Execute the Automated Test Suite (23 Tests)
Run the full pytest suite to verify all ingestion invariants, corporate-action adjustments, leak-free feature pipelines, and target alignments:
```bash
cd model_development
python -m pytest tests/ -v
```
*Expected Outcome: 23 passed in ~4.8 seconds.*

---

### Step 2: Run the Standalone Data Validation Audit
Audit all 16 raw CSV datasets, key uniqueness, OHLC pricing bounds, and fundamental data coverage:
```bash
cd model_development
python validate_data.py
```
*Expected Outcome: Generates `outputs/reports/data_quality_audit_report.json` with status PASS.*

---

### Step 3: Run the Ingestion, Feature & Target Pipeline
Process raw market Bhavcopy files, apply backward corporate action adjustments, compute 20 technical features, and construct forward targets:
```bash
cd model_development
python data_pipeline.py
```
*Expected Outcome: Creates `outputs/datasets/analytical_dataset.csv` (19,840 rows).*

---

### Step 4: Run the Training Pipeline Synthetic Smoke Test
Verify that all 5 classification models and all 5 regression models instantiate, fit, and evaluate properly on synthetic inputs:
```bash
cd model_development
python training_pipeline.py
```
*Expected Outcome: "Synthetic smoke test passed successfully for all 5 classification and 5 regression models."*

---

### Step 5: Run the Complete Master Quantitative Pipeline
Execute full walk-forward validation (3 chronological expanding folds with 5-session purge), evaluate on the untouched 15% holdout set, generate diagnostic figures, and save model artifacts:
```bash
cd model_development
python run_project.py
```
*Expected Outcome: Outputs all metrics, figures to `outputs/figures/`, and reports to `outputs/reports/`.*

---

### Step 6: Independently Verify Saved Holdout Metrics
Recalculate holdout metrics directly from raw predictions to guarantee zero hardcoding:
```bash
cd model_development
python verify_metrics.py
```
*Expected Outcome: Recalculated metrics match `model_comparison.csv` with diff < 10⁻¹⁶.*

---

### Step 7: Generate Directional Predictions (CLI)
Generate 5-day directional forecasts on the latest market data:
```bash
cd model_development

# Individual equity
python predict.py --symbol RELIANCE
python predict.py --symbol INFY

# Complete 16-stock universe
python predict.py --all
```

---

### Step 8: Launch the Interactive Streamlit Web Dashboard
Launch the multi-page web application from the project root or `model_development`:
```bash
# From project root:
streamlit run app.py

# Or from model_development:
cd model_development
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser.
