
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load prepared data
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = OUTPUT_DIR / "stock_features.csv"

data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
data = data.sort_values(["Date", "Symbol"]).reset_index(drop=True)

features = [
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

if "% Dly Qt to Traded Qty" in data.columns:
    features.append("% Dly Qt to Traded Qty")

missing = set(features + ["Date", "Symbol", "target_up_5d"]) - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

data["target_up_5d"] = pd.to_numeric(
    data["target_up_5d"], errors="coerce"
)
data = data.dropna(subset=["target_up_5d"]).copy()
data["target_up_5d"] = data["target_up_5d"].astype(int)

for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].replace([np.inf, -np.inf], np.nan)

dates = np.array(sorted(data["Date"].unique()))

# Three expanding-window validation folds.
# Each validation window is about 15% of all dates.
# Five full dates are excluded between train and validation.
PURGE_DATES = 5
fold_specs = [(0.50, 0.65), (0.65, 0.80), (0.80, 0.95)]

def make_models():
    return {
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
                random_state=42,
            )),
        ]),
        "Random_Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

# --------------------------------------------------
# 2. Run walk-forward validation
# --------------------------------------------------
results = []

for fold_no, (train_fraction, val_end_fraction) in enumerate(
    fold_specs, start=1
):
    train_end_idx = int(len(dates) * train_fraction) - 1
    val_end_idx = int(len(dates) * val_end_fraction) - 1
    val_start_idx = train_end_idx + PURGE_DATES + 1

    if val_start_idx >= val_end_idx:
        raise ValueError(f"Invalid date range for fold {fold_no}")

    train_end_date = pd.Timestamp(dates[train_end_idx])
    val_start_date = pd.Timestamp(dates[val_start_idx])
    val_end_date = pd.Timestamp(dates[val_end_idx])

    train = data[data["Date"] <= train_end_date]
    valid = data[
        (data["Date"] >= val_start_date)
        & (data["Date"] <= val_end_date)
    ]

    if train.empty or valid.empty:
        raise ValueError(f"Empty train/validation set in fold {fold_no}")

    X_train, y_train = train[features], train["target_up_5d"]
    X_valid, y_valid = valid[features], valid["target_up_5d"]

    print(f"\n{'=' * 55}")
    print(f"FOLD {fold_no}")
    print(f"Train through: {train_end_date.date()}")
    print(
        f"Validation: {val_start_date.date()} "
        f"to {val_end_date.date()}"
    )
    print(f"Train rows: {len(train):,}; validation rows: {len(valid):,}")

    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        result = {
            "fold": fold_no,
            "model": name,
            "train_end": str(train_end_date.date()),
            "validation_start": str(val_start_date.date()),
            "validation_end": str(val_end_date.date()),
            "train_rows": len(train),
            "validation_rows": len(valid),
            "accuracy": accuracy_score(y_valid, pred),
            "balanced_accuracy": balanced_accuracy_score(y_valid, pred),
            "f1": f1_score(y_valid, pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
                if y_valid.nunique() == 2 else np.nan
            ),
        }
        results.append(result)
        print(
            f"{name:24s} "
            f"Accuracy={result['accuracy']:.3f} "
            f"Balanced Acc={result['balanced_accuracy']:.3f} "
            f"F1={result['f1']:.3f} "
            f"ROC-AUC={result['roc_auc']:.3f}"
        )

# --------------------------------------------------
# 3. Save results
# --------------------------------------------------
result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_DIR / "walk_forward_results.csv", index=False
)

summary = (
    result_df.groupby("model")[
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    ]
    .agg(["mean", "std"])
    .round(4)
)

summary.to_csv(OUTPUT_DIR / "walk_forward_summary.csv")

print(f"\n{'=' * 55}")
print("WALK-FORWARD VALIDATION SUMMARY")
print(summary)
print(f"\nSaved: {OUTPUT_DIR / 'walk_forward_results.csv'}")
print(f"Saved: {OUTPUT_DIR / 'walk_forward_summary.csv'}")
