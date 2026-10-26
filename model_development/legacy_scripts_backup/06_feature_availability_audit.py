
from pathlib import Path
import json

import numpy as np
import pandas as pd

# =========================================================
# 1. Configuration
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_FILE = OUTPUT_DIR / "stock_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

required = {"Symbol", "Date", "Close Price", "target_up_5d"}
missing_required = required - set(df.columns)

if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# Features actually intended for training.
FEATURES = [
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

if "% Dly Qt to Traded Qty" in df.columns:
    FEATURES.append("% Dly Qt to Traded Qty")

missing_features = set(FEATURES) - set(df.columns)
if missing_features:
    raise ValueError(
        f"Training features absent from dataset: {missing_features}"
    )

# Parse numeric columns defensively.
numeric_columns = [
    "Close Price", "Open Price", "High Price", "Low Price",
    "Total Traded Quantity", "No. of Trades",
    "Prev Close", "Last Price", "Average Price",
    "Turnover ₹", "Deliverable Qty",
    "% Dly Qt to Traded Qty",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

for col in FEATURES + ["target_up_5d", "future_return_5d"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

g = df.groupby("Symbol", sort=False)

# =========================================================
# 2. Feature-name leakage checks
# =========================================================
# Only the explicit FEATURES list is eligible for model input.
# These names are never allowed as input features.
forbidden_words = (
    "target", "future", "forward", "label", "lead"
)

forbidden_features = [
    col for col in FEATURES
    if any(word in col.lower() for word in forbidden_words)
]

findings = []

def add_finding(check, status, severity, details):
    findings.append({
        "check": check,
        "status": status,
        "severity": severity,
        "details": str(details),
    })

add_finding(
    "Feature whitelist",
    "PASS" if not forbidden_features else "FAIL",
    "HIGH" if forbidden_features else "INFO",
    (
        "No target/future-named columns in the training feature list."
        if not forbidden_features
        else f"Forbidden feature names: {forbidden_features}"
    ),
)

# Check if future/target-related columns exist outside the whitelist.
suspicious_columns = [
    c for c in df.columns
    if any(word in c.lower() for word in forbidden_words)
    and c not in FEATURES
]

add_finding(
    "Outcome columns excluded from inputs",
    "PASS",
    "INFO",
    f"Outcome-related columns present but not selected: {suspicious_columns}",
)

# =========================================================
# 3. Availability at after-close prediction time
# =========================================================
# Raw daily OHLC and trading activity are known after the close,
# assuming the data source has finalized that session's records.
after_close_fields = [
    "Open Price", "High Price", "Low Price", "Close Price",
    "Prev Close", "Last Price", "Average Price",
    "Total Traded Quantity", "Turnover ₹", "No. of Trades",
    "Deliverable Qty", "% Dly Qt to Traded Qty",
]

present_fields = [c for c in after_close_fields if c in df.columns]
missing_fields = [c for c in after_close_fields if c not in df.columns]

add_finding(
    "After-close market data inventory",
    "INFO",
    "INFO",
    f"Present: {present_fields}; absent: {missing_fields}",
)

# =========================================================
# 4. Row keys and chronological order
# =========================================================
duplicate_count = int(df.duplicated(["Symbol", "Date"]).sum())

add_finding(
    "Unique stock-date keys",
    "PASS" if duplicate_count == 0 else "FAIL",
    "HIGH" if duplicate_count else "INFO",
    f"Duplicate rows: {duplicate_count}",
)

bad_order = 0
for _, group in df.groupby("Symbol", sort=False):
    if not group["Date"].is_monotonic_increasing:
        bad_order += 1

add_finding(
    "Chronological ordering",
    "PASS" if bad_order == 0 else "FAIL",
    "HIGH" if bad_order else "INFO",
    f"Stocks with unsorted dates: {bad_order}",
)

# =========================================================
# 5. Independently recompute core features
# =========================================================
# Comparison is performed only where both values exist.
formula_results = []

def compare_feature(name, expected, actual, tolerance=1e-6):
    expected = pd.Series(expected, index=df.index, dtype="float64")
    actual = pd.to_numeric(actual, errors="coerce").astype("float64")

    valid = expected.notna() & actual.notna()
    if valid.sum() == 0:
        formula_results.append({
            "feature": name,
            "compared_rows": 0,
            "mismatch_rows": None,
            "max_abs_error": None,
            "status": "NOT_TESTED",
        })
        return

    error = (expected[valid] - actual[valid]).abs()
    mismatch = error > tolerance

    formula_results.append({
        "feature": name,
        "compared_rows": int(valid.sum()),
        "mismatch_rows": int(mismatch.sum()),
        "max_abs_error": float(error.max()),
        "status": "PASS" if not mismatch.any() else "REVIEW",
    })

close = df["Close Price"]
close_g = df.groupby("Symbol", sort=False)["Close Price"]

# Past returns must only use current and previous closes.
for n in [1, 2, 5, 10, 20]:
    expected = close / close_g.shift(n) - 1
    compare_feature(f"return_{n}d", expected, df[f"return_{n}d"])

# Current day's open-close return is known after the close.
if "Open Price" in df.columns:
    expected = df["Close Price"] / df["Open Price"] - 1
    compare_feature(
        "open_close_return", expected, df["open_close_return"]
    )

# Volume change uses today's and the previous session's volume.
if "Total Traded Quantity" in df.columns:
    volume = df["Total Traded Quantity"]
    volume_g = df.groupby("Symbol", sort=False)[
        "Total Traded Quantity"
    ]
    expected = volume / volume_g.shift(1) - 1
    compare_feature(
        "volume_change_1d", expected, df["volume_change_1d"]
    )

# Close-to-SMA features may be encoded as close/SMA or close/SMA - 1.
# Determine which definition best matches the saved feature.
for n in [5, 10, 20, 50]:
    sma = close.groupby(df["Symbol"], sort=False).transform(
        lambda s: s.rolling(n, min_periods=n).mean()
    )
    ratio = close / sma
    relative_deviation = ratio - 1
    actual = df[f"close_vs_sma_{n}"]

    valid = actual.notna() & ratio.notna()
    if valid.sum() == 0:
        formula_results.append({
            "feature": f"close_vs_sma_{n}",
            "compared_rows": 0,
            "mismatch_rows": None,
            "max_abs_error": None,
            "status": "NOT_TESTED",
        })
        continue

    err_ratio = (actual[valid] - ratio[valid]).abs().median()
    err_deviation = (
        actual[valid] - relative_deviation[valid]
    ).abs().median()

    chosen = ratio if err_ratio <= err_deviation else relative_deviation
    definition = (
        "close/SMA" if err_ratio <= err_deviation
        else "close/SMA - 1"
    )

    compare_feature(f"close_vs_sma_{n}", chosen, actual)
    formula_results[-1]["matched_definition"] = definition

# =========================================================
# 6. OHLC consistency and impossible values
# =========================================================
ohlc_cols = {"Open Price", "High Price", "Low Price", "Close Price"}

if ohlc_cols.issubset(df.columns):
    o = df["Open Price"]
    h = df["High Price"]
    l = df["Low Price"]
    c = df["Close Price"]

    invalid_ohlc = (
        (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
    ) & o.notna() & h.notna() & l.notna() & c.notna()

    add_finding(
        "OHLC consistency",
        "PASS" if not invalid_ohlc.any() else "FAIL",
        "HIGH" if invalid_ohlc.any() else "INFO",
        f"Inconsistent rows: {int(invalid_ohlc.sum())}",
    )

if "Close Price" in df.columns:
    invalid_close = df["Close Price"].notna() & (df["Close Price"] <= 0)
    add_finding(
        "Positive closing prices",
        "PASS" if not invalid_close.any() else "FAIL",
        "HIGH" if invalid_close.any() else "INFO",
        f"Invalid rows: {int(invalid_close.sum())}",
    )

# =========================================================
# 7. Recompute the target independently
# =========================================================
future_close = close.groupby(df["Symbol"], sort=False).shift(-5)
has_future = future_close.notna()
has_target = df["target_up_5d"].notna()

expected_target = (future_close > close).astype("float64")
comparable = has_future & has_target

target_mismatches = (
    df.loc[comparable, "target_up_5d"].to_numpy()
    != expected_target.loc[comparable].to_numpy()
)

add_finding(
    "Five-row target alignment",
    "PASS" if target_mismatches.sum() == 0 else "FAIL",
    "HIGH" if target_mismatches.sum() else "INFO",
    (
        f"Compared rows: {int(comparable.sum())}; "
        f"mismatches: {int(target_mismatches.sum())}; "
        f"future price unavailable: {int((~has_future).sum())}"
    ),
)

# =========================================================
# 8. Missingness and availability warning
# =========================================================
missingness = pd.DataFrame({
    "feature": FEATURES,
    "missing_rows": [int(df[c].isna().sum()) for c in FEATURES],
    "missing_percent": [
        round(float(df[c].isna().mean() * 100), 3) for c in FEATURES
    ],
})

add_finding(
    "Fundamental-data coverage",
    "INFO",
    "MEDIUM",
    (
        "No fundamental features are included in the current training "
        "feature list. This is a market-data-only experiment."
    ),
)

add_finding(
    "Corporate-action adjustment",
    "NOT_VERIFIED",
    "HIGH",
    (
        "This CSV alone does not establish whether prices are adjusted "
        "for splits, bonuses, dividends, or other corporate actions. "
        "Verify the data vendor's adjustment policy separately."
    ),
)

add_finding(
    "Point-in-time feature availability",
    "PARTIAL",
    "HIGH",
    (
        "After-close OHLC and trading activity are eligible if the model "
        "prediction is made after those values are finalized. This script "
        "cannot verify original publication timestamps or vendor revisions."
    ),
)

# =========================================================
# 9. Save reproducible outputs
# =========================================================
formula_df = pd.DataFrame(formula_results)
findings_df = pd.DataFrame(findings)

formula_path = OUTPUT_DIR / "feature_formula_audit.csv"
findings_path = OUTPUT_DIR / "feature_availability_audit.csv"
missingness_path = OUTPUT_DIR / "feature_missingness_audit.csv"
json_path = OUTPUT_DIR / "feature_audit_summary.json"

formula_df.to_csv(formula_path, index=False)
findings_df.to_csv(findings_path, index=False)
missingness.to_csv(missingness_path, index=False)

summary = {
    "prediction_time": "after market close",
    "rows": int(len(df)),
    "symbols": int(df["Symbol"].nunique()),
    "features_audited": FEATURES,
    "formula_checks": formula_results,
    "findings": findings,
    "interpretation": (
        "A PASS only means the specific programmed check passed. "
        "Corporate-action adjustments, vendor publication timing, "
        "and all upstream feature-generation assumptions require "
        "separate verification."
    ),
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, allow_nan=False)

print("\n" + "=" * 60)
print("FEATURE AVAILABILITY & LEAKAGE AUDIT")
print("=" * 60)

print("\nGeneral checks:")
print(findings_df.to_string(index=False))

print("\nFormula checks:")
print(formula_df.to_string(index=False))

print("\nFeature missingness:")
print(missingness.to_string(index=False))

print("\nSaved reports:")
print(findings_path)
print(formula_path)
print(missingness_path)
print(json_path)
