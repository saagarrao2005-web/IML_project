
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = OUTPUT_DIR / "stock_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FRACTION = 0.20
PURGE_SESSIONS = 5
RANDOM_STATE = 42

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
data = data.sort_values(["Date", "Symbol"]).reset_index(drop=True)

required = {"Date", "Symbol", "target_up_5d"}
missing_columns = required - set(data.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# --------------------------------------------------
# 2. Select features explicitly
#    Exclude future outcomes and raw price columns.
feature_cols = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "close_vs_sma_5",
    "close_vs_sma_10",
    "close_vs_sma_20",
    "close_vs_sma_50",
    "volatility_10d",
    "volatility_20d",
    "open_close_return",
    "volume_change_1d",
    "volume_vs_avg20",
    "trades_vs_avg20",
    "day_of_week",
    "month",
]

# Add delivery data only if it exists in this prepared dataset.
# It may be missing on some days, so the training pipeline imputes it.
if "% Dly Qt to Traded Qty" in data.columns:
    feature_cols.append("% Dly Qt to Traded Qty")

missing_features = set(feature_cols) - set(data.columns)
if missing_features:
    raise ValueError(f"Feature columns not found: {missing_features}")

# Ensure target is numeric and retain only rows with known labels.
data["target_up_5d"] = pd.to_numeric(
    data["target_up_5d"], errors="coerce"
)

data = data.dropna(subset=["target_up_5d"]).copy()
data["target_up_5d"] = data["target_up_5d"].astype(int)

if not set(data["target_up_5d"].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

# Keep only finite numeric feature values; remaining NaNs are imputed
# inside each training pipeline to avoid fitting the imputer on test data.
for col in feature_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].replace([np.inf, -np.inf], np.nan)

print(f"Rows with labels: {len(data):,}")
print(f"Stocks: {data['Symbol'].nunique()}")
print(f"Features: {len(feature_cols)}")
print(f"Date range: {data['Date'].min().date()} to "
      f"{data['Date'].max().date()}")

# --------------------------------------------------
# 3. Chronological train/test split by distinct dates
# --------------------------------------------------
dates = np.array(sorted(data["Date"].dropna().unique()))

if len(dates) < 30:
    raise ValueError("Too few distinct dates for a reliable time split.")

split_idx = int(len(dates) * (1 - TEST_FRACTION))

# Purge a five-session buffer immediately before test starts.
test_start_idx = split_idx + PURGE_SESSIONS

if test_start_idx >= len(dates):
    raise ValueError("Not enough dates after applying the purge window.")

test_start_date = pd.Timestamp(dates[test_start_idx])
train_end_date = pd.Timestamp(dates[split_idx - 1])

train = data[data["Date"] <= train_end_date].copy()
test = data[data["Date"] >= test_start_date].copy()

# Remove any overlap in dates and ensure every stock follows the same
# calendar split. The purge creates a gap between train and test.
assert train["Date"].max() < test["Date"].min()

if train.empty or test.empty:
    raise ValueError("Train/test split produced an empty dataset.")

X_train = train[feature_cols]
y_train = train["target_up_5d"]

X_test = test[feature_cols]
y_test = test["target_up_5d"]

print("\nChronological split")
print(f"Train: {len(train):,} rows through "
      f"{train['Date'].max().date()}")
print(f"Test:  {len(test):,} rows from "
      f"{test['Date'].min().date()}")
print(f"Purged dates: {PURGE_SESSIONS}")
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).sort_index().round(3))
print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).sort_index().round(3))

# --------------------------------------------------
# 4. Define models
# --------------------------------------------------
models = {
    "Dummy_Baseline": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent")),
    ]),

    "Logistic_Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),

    "Random_Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
}

# --------------------------------------------------
# 5. Train and evaluate
# --------------------------------------------------
results = []
reports = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
    else:
        probabilities = None
        metrics["roc_auc"] = np.nan

    results.append(metrics)

    reports[name] = {
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "test_start": str(test["Date"].min().date()),
        "test_end": str(test["Date"].max().date()),
    }

    print(pd.Series(metrics).to_string())
    print("Confusion matrix [actual rows 0,1; predicted columns 0,1]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# --------------------------------------------------
# 6. Save comparison and evaluation reports
# --------------------------------------------------
results_df = pd.DataFrame(results).sort_values(
    "balanced_accuracy", ascending=False
)

results_path = OUTPUT_DIR / "model_comparison.csv"
report_path = OUTPUT_DIR / "evaluation_report.json"

results_df.to_csv(results_path, index=False)

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(reports, f, indent=2, allow_nan=True)

# Choose the model with the best balanced accuracy.
# This is a preliminary comparison, not final model selection:
# robust selection should use time-series validation.
best_name = results_df.iloc[0]["model"]
best_model = models[best_name]

model_path = OUTPUT_DIR / "best_model.joblib"
joblib.dump(
    {
        "model": best_model,
        "feature_columns": feature_cols,
        "model_name": best_name,
        "target_definition": (
            "1 if close price five rows later exceeds current close; "
            "0 otherwise"
        ),
        "test_start": str(test["Date"].min().date()),
        "test_end": str(test["Date"].max().date()),
    },
    model_path,
)

print("\n" + "=" * 55)
print("MODEL TRAINING COMPLETED")
print("=" * 55)
print("\nComparison:")
print(results_df.to_string(index=False))
print(f"\nComparison CSV: {results_path}")
print(f"Evaluation report: {report_path}")
print(f"Saved best model: {model_path}")
print(
    "\nImportant: treat this as a baseline experiment. "
    "Do not interpret accuracy as guaranteed trading profitability."
)
