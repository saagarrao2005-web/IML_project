"""Independent metric verification script."""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)

# Reload holdout predictions and recalculate all metrics independently
preds = pd.read_csv("outputs/predictions/holdout_predictions.csv", parse_dates=["Date"])
comp = pd.read_csv("outputs/reports/model_comparison.csv")

y_true = preds["target_up_5d"].astype(int).to_numpy()
print(f"Holdout N={len(preds)}, class_1_ratio={y_true.mean():.4f}")

# Recalculate for each model
for model in ["Dummy_Baseline", "Logistic_Regression", "Random_Forest",
              "Hist_Gradient_Boosting", "XGBoost"]:
    prob_col = f"{model}_prob"
    pred_col = f"{model}_pred"
    if prob_col not in preds.columns:
        print(f"{model}: prob column missing")
        continue
    probs = preds[prob_col].to_numpy()
    preds_cls = preds[pred_col].astype(int).to_numpy()

    recalc_auc = roc_auc_score(y_true, probs)
    recalc_acc = accuracy_score(y_true, preds_cls)
    recalc_bacc = balanced_accuracy_score(y_true, preds_cls)
    recalc_prec = precision_score(y_true, preds_cls, zero_division=0)
    recalc_rec = recall_score(y_true, preds_cls, zero_division=0)
    recalc_f1 = f1_score(y_true, preds_cls, zero_division=0)

    saved = comp[comp["model"] == model].iloc[0]
    auc_diff = abs(recalc_auc - saved["roc_auc"])
    acc_diff = abs(recalc_acc - saved["accuracy"])

    status = "MATCH" if auc_diff < 1e-10 and acc_diff < 1e-10 else "MISMATCH"
    print(f"  {model}:")
    print(f"    AUC:  recalc={recalc_auc:.10f}  saved={saved['roc_auc']:.10f}  diff={auc_diff:.2e}  {status}")
    print(f"    Acc:  recalc={recalc_acc:.10f}  saved={saved['accuracy']:.10f}  diff={acc_diff:.2e}")
    print(f"    BAcc: recalc={recalc_bacc:.10f}  saved={saved['balanced_accuracy']:.10f}")
    print(f"    Prec: {recalc_prec:.6f}  Rec: {recalc_rec:.6f}  F1: {recalc_f1:.6f}")

print()

# Verify chronological separation
print(f"Holdout date range: {preds['Date'].min().date()} to {preds['Date'].max().date()}")

df = pd.read_csv("outputs/datasets/analytical_dataset.csv", parse_dates=["Date"])
labeled = df.dropna(subset=["target_up_5d"])
all_dates = np.array(sorted(labeled["Date"].unique()))
n_dates = len(all_dates)
split_idx = int(n_dates * 0.85)
dev_max = pd.Timestamp(all_dates[split_idx - 1])
holdout_start_idx = split_idx + 5
holdout_min_date = pd.Timestamp(all_dates[holdout_start_idx])
gap = holdout_start_idx - split_idx
print(f"Dev max date: {dev_max.date()}")
print(f"Holdout min date: {holdout_min_date.date()}")
print(f"Purge gap: {gap} trading sessions")
print(f"Calendar gap: {(holdout_min_date - dev_max).days} days")

# Check target uses only future within same stock
print()
print("Target alignment spot-check (RELIANCE):")
rel = df[df["Symbol"] == "RELIANCE"].sort_values("Date").reset_index(drop=True)
mismatches = 0
for i in range(min(50, len(rel) - 5)):
    actual_future = rel.iloc[i + 5]["adj_close"]
    stored_future = rel.iloc[i]["future_adj_close_5d"]
    if not np.isclose(actual_future, stored_future, rtol=1e-10):
        mismatches += 1
        print(f"  MISMATCH at row {i}")
print(f"  Checked 50 rows, {mismatches} mismatches")

# Verify last 5 per stock have NaN
all_stocks_ok = True
for sym in df["Symbol"].unique():
    s = df[df["Symbol"] == sym].sort_values("Date")
    if not s["target_up_5d"].tail(5).isna().all():
        print(f"  ERROR: {sym} does not have 5 NaN targets at end")
        all_stocks_ok = False
print(f"  All stocks have correct NaN tail targets: {all_stocks_ok}")
