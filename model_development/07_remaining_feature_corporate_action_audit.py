
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# 07 - REMAINING FEATURE & CORPORATE-ACTION AUDIT
# ============================================================

BASE_DIR = Path(
    r"D:\B.Tech\Sem3\IML\Project\Stock_Prediction\model_development"
)
INPUT_FILE = BASE_DIR / "outputs" / "stock_features.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Large daily moves are REVIEW candidates, not proof of an error.
LARGE_MOVE_THRESHOLD = 0.15
FORMULA_TOLERANCE = 1e-8

print("=" * 68)
print("REMAINING FEATURE & CORPORATE-ACTION AUDIT")
print("=" * 68)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Check the path and run 02_prepare_features.py first."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

required = [
    "Symbol", "Date", "Open Price", "High Price", "Low Price",
    "Prev Close", "Close Price", "Total Traded Quantity",
    "No. of Trades", "future_return_5d", "target_up_5d",
]

missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        "Check that you are using outputs/stock_features.csv."
    )

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = [
    "Open Price", "High Price", "Low Price", "Prev Close",
    "Close Price", "Total Traded Quantity", "No. of Trades",
    "future_return_5d", "target_up_5d",
    "volatility_10d", "volatility_20d",
    "volume_vs_avg20", "trades_vs_avg20",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
groups = df.groupby("Symbol", sort=False)

print(f"Input: {INPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Stocks: {df['Symbol'].nunique()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")


# ------------------------------------------------------------
# 1. RECOMPUTE THE REMAINING FEATURES
# ------------------------------------------------------------

close = groups["Close Price"]
volume = groups["Total Traded Quantity"]
trades = groups["No. of Trades"]

# One-day close-to-close return, independently recomputed.
df["_audit_return_1d"] = close.pct_change(fill_method=None)

# Rolling sample standard deviation of daily returns.
df["_expected_volatility_10d"] = (
    df.groupby("Symbol", sort=False)["_audit_return_1d"]
      .transform(lambda s: s.rolling(10, min_periods=10).std())
)

df["_expected_volatility_20d"] = (
    df.groupby("Symbol", sort=False)["_audit_return_1d"]
      .transform(lambda s: s.rolling(20, min_periods=20).std())
)

# Rolling averages exclude no observations from the current day:
# at after-close prediction time, today's volume/trades are available.
df["_avg_volume_20d"] = volume.transform(
    lambda s: s.rolling(20, min_periods=20).mean()
)
df["_avg_trades_20d"] = trades.transform(
    lambda s: s.rolling(20, min_periods=20).mean()
)

df["_volume_ratio"] = (
    df["Total Traded Quantity"] / df["_avg_volume_20d"]
)
df["_trades_ratio"] = (
    df["No. of Trades"] / df["_avg_trades_20d"]
)


# ------------------------------------------------------------
# 2. FORMULA AND MISSINGNESS AUDIT
# ------------------------------------------------------------

formula_results = []


def compare_formula(feature, expected, description):
    """Compare saved values with independently computed values."""
    if feature not in df.columns:
        formula_results.append({
            "feature": feature,
            "definition_tested": description,
            "compared_rows": 0,
            "mismatch_rows": None,
            "max_abs_error": None,
            "missing_mask_mismatches": None,
            "status": "MISSING_FEATURE",
        })
        return

    actual = pd.to_numeric(df[feature], errors="coerce")
    expected = pd.Series(expected, index=df.index, dtype="float64")

    actual_missing = actual.isna()
    expected_missing = expected.isna()
    mask_mismatches = int((actual_missing != expected_missing).sum())

    valid = actual.notna() & expected.notna()
    errors = (actual[valid] - expected[valid]).abs()

    mismatch_rows = int((errors > FORMULA_TOLERANCE).sum())
    max_error = float(errors.max()) if len(errors) else None

    if mask_mismatches > 0:
        status = "REVIEW_MISSINGNESS"
    elif mismatch_rows > 0:
        status = "REVIEW_FORMULA"
    elif len(errors) == 0:
        status = "NO_COMPARABLE_ROWS"
    else:
        status = "PASS"

    formula_results.append({
        "feature": feature,
        "definition_tested": description,
        "compared_rows": int(valid.sum()),
        "mismatch_rows": mismatch_rows,
        "max_abs_error": max_error,
        "missing_mask_mismatches": mask_mismatches,
        "status": status,
    })


compare_formula(
    "volatility_10d",
    df["_expected_volatility_10d"],
    "Rolling 10-observation sample standard deviation of daily returns",
)

compare_formula(
    "volatility_20d",
    df["_expected_volatility_20d"],
    "Rolling 20-observation sample standard deviation of daily returns",
)


def compare_either_definition(feature, ratio, description):
    """Test ratio and ratio-minus-one without assuming which was saved."""
    if feature not in df.columns:
        compare_formula(feature, ratio, description + ": ratio")
        return

    actual = pd.to_numeric(df[feature], errors="coerce")
    candidates = [
        ("ratio", pd.Series(ratio, index=df.index, dtype="float64")),
        (
            "ratio_minus_1",
            pd.Series(ratio, index=df.index, dtype="float64") - 1,
        ),
    ]

    best = None

    for name, expected in candidates:
        valid = actual.notna() & expected.notna()
        if not valid.any():
            continue

        errors = (actual[valid] - expected[valid]).abs()
        mismatches = int((errors > FORMULA_TOLERANCE).sum())
        mask_mismatches = int(
            (actual.isna() != expected.isna()).sum()
        )
        score = (mask_mismatches, mismatches)

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "name": name,
                "expected": expected,
                "valid": valid,
                "errors": errors,
                "mismatches": mismatches,
                "mask_mismatches": mask_mismatches,
            }

    if best is None:
        formula_results.append({
            "feature": feature,
            "definition_tested": description,
            "compared_rows": 0,
            "mismatch_rows": None,
            "max_abs_error": None,
            "missing_mask_mismatches": None,
            "status": "NO_COMPARABLE_ROWS",
        })
        return

    formula_results.append({
        "feature": feature,
        "definition_tested": f"{description}: {best['name']}",
        "compared_rows": int(best["valid"].sum()),
        "mismatch_rows": best["mismatches"],
        "max_abs_error": (
            float(best["errors"].max())
            if len(best["errors"]) else None
        ),
        "missing_mask_mismatches": best["mask_mismatches"],
        "status": (
            "REVIEW_MISSINGNESS"
            if best["mask_mismatches"] > 0
            else "REVIEW_FORMULA"
            if best["mismatches"] > 0
            else "PASS"
        ),
    })


compare_either_definition(
    "volume_vs_avg20",
    df["_volume_ratio"],
    "Current volume divided by 20-observation rolling mean",
)

compare_either_definition(
    "trades_vs_avg20",
    df["_trades_ratio"],
    "Current trade count divided by 20-observation rolling mean",
)

formula_df = pd.DataFrame(formula_results)
formula_path = OUTPUT_DIR / "remaining_feature_formula_audit.csv"
formula_df.to_csv(formula_path, index=False)

print("\nRemaining-feature formula audit:")
print(formula_df.to_string(index=False))


# ------------------------------------------------------------
# 3. CHECK PRICE MOVES THAT MAY NEED CORPORATE-ACTION REVIEW
# ------------------------------------------------------------

df["_previous_close_in_file"] = groups["Close Price"].shift(1)

# Compare with the previous row's close and the supplied NSE Prev Close.
df["_return_from_previous_row"] = (
    df["Close Price"] / df["_previous_close_in_file"] - 1
)

df["_return_from_prev_close"] = (
    df["Close Price"] / df["Prev Close"] - 1
)

# Price ratio is useful for finding factor-like moves, but it is
# NOT sufficient to identify a split or bonus issue.
df["_close_ratio_to_previous"] = (
    df["Close Price"] / df["_previous_close_in_file"]
)

df["_factor_like_move"] = False

for factor in [2, 3, 4, 5, 10, 0.5, 1/3, 0.25, 0.2, 0.1]:
    df["_factor_like_move"] |= (
        np.isfinite(df["_close_ratio_to_previous"])
        & (
            (df["_close_ratio_to_previous"] / factor - 1).abs()
            <= 0.05
        )
    )

large_move = (
    df["_return_from_previous_row"].abs() >= LARGE_MOVE_THRESHOLD
)
prev_close_move = (
    df["_return_from_prev_close"].abs() >= LARGE_MOVE_THRESHOLD
)

candidate_mask = (
    large_move.fillna(False)
    | prev_close_move.fillna(False)
    | df["_factor_like_move"].fillna(False)
)

candidates = df.loc[candidate_mask].copy()

if candidates.empty:
    candidates = pd.DataFrame(columns=[
        "Symbol", "Date", "Open Price", "High Price", "Low Price",
        "Prev Close", "Close Price", "Total Traded Quantity",
        "_previous_close_in_file", "_return_from_previous_row",
        "_return_from_prev_close", "_close_ratio_to_previous",
        "_factor_like_move",
    ])

candidates["review_reason"] = ""

for idx in candidates.index:
    reasons = []
    row = candidates.loc[idx]

    r1 = row.get("_return_from_previous_row", np.nan)
    r2 = row.get("_return_from_prev_close", np.nan)

    if pd.notna(r1) and abs(r1) >= LARGE_MOVE_THRESHOLD:
        reasons.append("Large close-to-close move")

    if pd.notna(r2) and abs(r2) >= LARGE_MOVE_THRESHOLD:
        reasons.append("Large move vs supplied Prev Close")

    if bool(row.get("_factor_like_move", False)):
        reasons.append("Price ratio near common split/bonus factor")

    candidates.at[idx, "review_reason"] = "; ".join(reasons)

candidate_columns = [
    "Symbol", "Date", "Open Price", "High Price", "Low Price",
    "Prev Close", "Close Price", "Total Traded Quantity",
    "_previous_close_in_file", "_return_from_previous_row",
    "_return_from_prev_close", "_close_ratio_to_previous",
    "_factor_like_move", "review_reason",
]
candidate_columns = [
    c for c in candidate_columns if c in candidates.columns
]

candidates = candidates[candidate_columns]
candidate_path = OUTPUT_DIR / "corporate_action_candidates.csv"
candidates.to_csv(candidate_path, index=False)

print("\nPotential corporate-action / large-move candidates:")
print(f"Candidate rows: {len(candidates)}")
if len(candidates):
    print(candidates.head(30).to_string(index=False))
else:
    print("No candidates met the configured thresholds.")


# ------------------------------------------------------------
# 4. VERIFY FIVE-OBSERVATION TARGET AGAINST CLOSE PRICES
# ------------------------------------------------------------

df["_expected_future_close_5"] = (
    groups["Close Price"].shift(-5)
)

df["_expected_future_return_5"] = (
    df["_expected_future_close_5"] / df["Close Price"] - 1
)

df["_expected_target_up_5"] = (
    df["_expected_future_return_5"] > 0
).astype("float64")

# No future price means no valid label should be available.
has_future = df["_expected_future_close_5"].notna()
has_saved_return = df["future_return_5d"].notna()
has_saved_target = df["target_up_5d"].notna()

return_valid = has_future & has_saved_return
target_valid = has_future & has_saved_target

return_mismatches = int((
    (
        df.loc[return_valid, "future_return_5d"]
        - df.loc[return_valid, "_expected_future_return_5"]
    ).abs() > FORMULA_TOLERANCE
).sum())

target_mismatches = int((
    df.loc[target_valid, "target_up_5d"].astype(float)
    != df.loc[target_valid, "_expected_target_up_5"]
).sum())

unexpected_labels = int((~has_future & has_saved_target).sum())
missing_labels_when_future_exists = int(
    (has_future & (~has_saved_target | ~has_saved_return)).sum()
)

target_summary = {
    "rows": int(len(df)),
    "rows_with_future_close_5": int(has_future.sum()),
    "future_return_compared_rows": int(return_valid.sum()),
    "future_return_mismatches": return_mismatches,
    "target_compared_rows": int(target_valid.sum()),
    "target_mismatches": target_mismatches,
    "labels_present_without_future_close": unexpected_labels,
    "missing_labels_when_future_close_exists": (
        missing_labels_when_future_exists
    ),
}

print("\nFive-observation target recheck:")
print(json.dumps(target_summary, indent=2))


# ------------------------------------------------------------
# 5. AUDIT ROLLING-FEATURE MISSINGNESS
# ------------------------------------------------------------

missingness_results = []

for feature in [
    "volatility_10d",
    "volatility_20d",
    "volume_vs_avg20",
    "trades_vs_avg20",
]:
    if feature not in df.columns:
        continue

    actual = pd.to_numeric(df[feature], errors="coerce")
    missingness_results.append({
        "feature": feature,
        "missing_rows": int(actual.isna().sum()),
        "missing_percent": round(float(actual.isna().mean() * 100), 4),
        "expected_rows_missing_from_lookback": (
            int(df["_expected_volatility_10d"].isna().sum())
            if feature == "volatility_10d"
            else int(df["_expected_volatility_20d"].isna().sum())
            if feature == "volatility_20d"
            else int(df["_volume_ratio"].isna().sum())
            if feature == "volume_vs_avg20"
            else int(df["_trades_ratio"].isna().sum())
        ),
    })

missingness_df = pd.DataFrame(missingness_results)
missingness_path = OUTPUT_DIR / "remaining_feature_missingness_audit.csv"
missingness_df.to_csv(missingness_path, index=False)

print("\nRolling-feature missingness:")
print(missingness_df.to_string(index=False))


# ------------------------------------------------------------
# 6. SUMMARY AND LIMITATIONS
# ------------------------------------------------------------

formula_passes = int((formula_df["status"] == "PASS").sum())
formula_reviews = int(
    formula_df["status"].astype(str).str.startswith("REVIEW").sum()
)

summary = {
    "input_file": str(INPUT_FILE),
    "rows": int(len(df)),
    "unique_symbols": int(df["Symbol"].nunique()),
    "duplicate_symbol_date_rows": int(
        df.duplicated(["Symbol", "Date"]).sum()
    ),
    "remaining_feature_checks": formula_results,
    "formula_checks_passed": formula_passes,
    "formula_checks_needing_review": formula_reviews,
    "corporate_action_candidate_rows": int(len(candidates)),
    "target_recheck": target_summary,
    "corporate_action_adjustment_status": (
        "UNVERIFIED: confirm the NSE data fields and any corporate-action "
        "adjustments against official event records."
    ),
    "point_in_time_status": (
        "PARTIAL: this script checks historical formulas and labels but "
        "cannot prove when vendor data became available or whether records "
        "were later revised."
    ),
    "interpretation": [
        "A large move is a review candidate, not proof of a corporate action.",
        "No automatic price adjustment or row deletion was performed.",
        "Check official NSE records for splits, bonuses, dividends and "
        "other relevant events before changing the dataset.",
        "If prices are unadjusted, verify whether the chosen target and "
        "features should use raw prices, adjusted prices, or both."
    ],
}

summary_path = OUTPUT_DIR / "remaining_feature_corporate_action_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, allow_nan=False, default=str)

print("\n" + "=" * 68)
print("AUDIT COMPLETE")
print("=" * 68)
print(f"Formula report:       {formula_path}")
print(f"Missingness report:   {missingness_path}")
print(f"Price-move candidates:{candidate_path}")
print(f"Summary JSON:         {summary_path}")
print("\nNo source data was modified.")
