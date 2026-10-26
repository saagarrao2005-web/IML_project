
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "stock_features.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

required = {"Symbol", "Date", "Close Price", "target_up_5d"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Close Price"] = pd.to_numeric(
    df["Close Price"].astype(str).str.replace(",", "", regex=False),
    errors="coerce",
)
df["target_up_5d"] = pd.to_numeric(
    df["target_up_5d"], errors="coerce"
)

# Recompute the five-row forward close independently for each stock.
df["recomputed_future_close_5d"] = (
    df.groupby("Symbol", sort=False)["Close Price"].shift(-5)
)

has_future = df["recomputed_future_close_5d"].notna()
has_target = df["target_up_5d"].notna()

expected_target = (
    df["recomputed_future_close_5d"] > df["Close Price"]
).astype("float64")

comparable = has_future & has_target

target_mismatches = (
    df.loc[comparable, "target_up_5d"].to_numpy()
    != expected_target.loc[comparable].to_numpy()
)

missing_target_with_future = has_future & ~has_target
target_without_future = ~has_future & has_target

duplicate_keys = df.duplicated(["Symbol", "Date"], keep=False)

# Check the ordering and gaps in dates for each stock.
date_gap_reports = []
for symbol, group in df.groupby("Symbol"):
    dates = group["Date"].sort_values()
    date_gap_reports.append({
        "Symbol": symbol,
        "rows": len(group),
        "duplicate_dates": int(dates.duplicated().sum()),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
    })

summary = {
    "total_rows": int(len(df)),
    "unique_symbols": int(df["Symbol"].nunique()),
    "duplicate_symbol_date_rows": int(duplicate_keys.sum()),
    "rows_with_future_close": int(has_future.sum()),
    "rows_with_target": int(has_target.sum()),
    "target_mismatches": int(target_mismatches.sum()),
    "missing_target_but_future_exists": int(
        missing_target_with_future.sum()
    ),
    "target_present_but_future_missing": int(
        target_without_future.sum()
    ),
    "expected_unavailable_future_rows": int((~has_future).sum()),
}

print("=" * 55)
print("TARGET AND ALIGNMENT AUDIT")
print("=" * 55)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nPer-stock date coverage:")
coverage = pd.DataFrame(date_gap_reports)
print(coverage.to_string(index=False))

# Save row-level exceptions for investigation.
exceptions = df.loc[
    duplicate_keys
    | (has_future & has_target & (
        df["target_up_5d"] != expected_target
    ))
    | missing_target_with_future
    | target_without_future
].copy()

exception_path = OUTPUT_DIR / "target_audit_exceptions.csv"
summary_path = OUTPUT_DIR / "target_audit_summary.csv"

exceptions.to_csv(exception_path, index=False)
pd.DataFrame([summary]).to_csv(summary_path, index=False)

print(f"\nExceptions: {exception_path}")
print(f"Summary: {summary_path}")

if summary["target_mismatches"] > 0:
    print("\nWARNING: Target labels do not match recomputed forward closes.")

if summary["missing_target_but_future_exists"] > 0:
    print("\nWARNING: Some rows have future prices but missing labels.")

if summary["target_present_but_future_missing"] > 0:
    print("\nWARNING: Some labels exist without a five-row future close.")
