
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# 08 - CORPORATE-ACTION SOURCE AUDIT
# Reads original cleaned NSE CSV files and prepared features.
# Does NOT modify original data or adjust prices.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "outputs" / "equity_series_only"
FEATURES_FILE = BASE_DIR / "outputs" / "stock_features.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LARGE_MOVE_THRESHOLD = 0.15
FACTOR_TOLERANCE = 0.05

print("=" * 72)
print("CORPORATE-ACTION SOURCE AUDIT")
print("=" * 72)
print(f"Original CSV folder: {RAW_DIR}")
print(f"Prepared features:   {FEATURES_FILE}")

if not RAW_DIR.exists():
    raise FileNotFoundError(f"Original data folder not found: {RAW_DIR}")

if not FEATURES_FILE.exists():
    raise FileNotFoundError(f"Prepared dataset not found: {FEATURES_FILE}")

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found directly inside {RAW_DIR}"
    )


def normalize_columns(df):
    """Normalize column names for matching without changing source files."""
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]
    return df


def find_column(df, possible_names):
    """Find a column using case-insensitive exact-name matching."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def to_numeric(series):
    """Convert common NSE numeric formats to numeric values."""
    s = series.astype("string").str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("₹", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.replace({
        "": pd.NA,
        "-": pd.NA,
        "--": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "null": pd.NA,
    })
    return pd.to_numeric(s, errors="coerce")


def infer_symbol(df, path):
    col = find_column(df, ["Symbol", "SYMBOL", "Ticker"])
    if col is not None:
        values = df[col].dropna().astype(str).str.strip().unique()
        if len(values) == 1:
            return values[0].upper()
        if len(values) > 1:
            raise ValueError(
                f"{path.name} contains multiple symbols: {values[:10]}"
            )
    return path.stem.strip().upper()


# ------------------------------------------------------------
# 1. LOAD ORIGINAL CSV FILES
# ------------------------------------------------------------

frames = []
file_inventory = []

for path in csv_files:
    try:
        raw = pd.read_csv(path, low_memory=False)
        raw = normalize_columns(raw)

        date_col = find_column(raw, ["Date", "Trade Date", "Timestamp"])
        close_col = find_column(
            raw, ["Close Price", "Close", "Closing Price"]
        )
        prev_close_col = find_column(
            raw, ["Prev Close", "Previous Close", "Prev. Close"]
        )
        open_col = find_column(raw, ["Open Price", "Open"])
        high_col = find_column(raw, ["High Price", "High"])
        low_col = find_column(raw, ["Low Price", "Low"])

        if date_col is None or close_col is None:
            file_inventory.append({
                "file": path.name,
                "status": "SKIPPED",
                "rows": len(raw),
                "reason": (
                    f"Could not identify date/close columns. "
                    f"Columns: {list(raw.columns)}"
                ),
            })
            continue

        symbol = infer_symbol(raw, path)

        out = pd.DataFrame({
            "Symbol": symbol,
            "Date": pd.to_datetime(raw[date_col], errors="coerce"),
            "raw_close": to_numeric(raw[close_col]),
        })

        out["raw_prev_close"] = (
            to_numeric(raw[prev_close_col])
            if prev_close_col is not None else np.nan
        )
        out["raw_open"] = (
            to_numeric(raw[open_col])
            if open_col is not None else np.nan
        )
        out["raw_high"] = (
            to_numeric(raw[high_col])
            if high_col is not None else np.nan
        )
        out["raw_low"] = (
            to_numeric(raw[low_col])
            if low_col is not None else np.nan
        )
        out["source_file"] = path.name

        invalid_dates = int(out["Date"].isna().sum())
        duplicate_keys = int(
            out.duplicated(["Symbol", "Date"]).sum()
        )

        frames.append(out)

        file_inventory.append({
            "file": path.name,
            "symbol": symbol,
            "status": "LOADED",
            "rows": len(out),
            "invalid_dates": invalid_dates,
            "duplicate_symbol_date_rows": duplicate_keys,
            "reason": "",
        })

    except Exception as exc:
        file_inventory.append({
            "file": path.name,
            "status": "ERROR",
            "rows": None,
            "reason": str(exc),
        })

if not frames:
    raise ValueError(
        "No usable original CSV files were loaded. "
        "Check the date and close column names."
    )

original = pd.concat(frames, ignore_index=True)
original = original.dropna(subset=["Date"])
original["Symbol"] = original["Symbol"].astype(str).str.strip().str.upper()
original = original.sort_values(["Symbol", "Date"]).reset_index(drop=True)

inventory_df = pd.DataFrame(file_inventory)
inventory_path = OUTPUT_DIR / "source_file_inventory.csv"
inventory_df.to_csv(inventory_path, index=False)

print(f"\nCSV files discovered: {len(csv_files)}")
print(f"Files loaded: {(inventory_df['status'] == 'LOADED').sum()}")
print(f"Original rows loaded: {len(original):,}")
print(f"Symbols loaded: {original['Symbol'].nunique()}")


# ------------------------------------------------------------
# 2. LOAD PREPARED FEATURES AND MATCH RAW PRICES
# ------------------------------------------------------------

features = pd.read_csv(FEATURES_FILE, low_memory=False)
features = normalize_columns(features)

required_feature_cols = ["Symbol", "Date", "Close Price"]

missing = [c for c in required_feature_cols if c not in features.columns]
if missing:
    raise ValueError(
        f"Prepared dataset is missing columns: {missing}"
    )

features["Symbol"] = (
    features["Symbol"].astype(str).str.strip().str.upper()
)
features["Date"] = pd.to_datetime(features["Date"], errors="coerce")
features["Close Price"] = to_numeric(features["Close Price"])

if features.duplicated(["Symbol", "Date"]).any():
    raise ValueError(
        "Prepared features contain duplicate Symbol-Date rows. "
        "Resolve duplicates before proceeding."
    )

original_match = original[
    [
        "Symbol", "Date", "raw_close", "raw_prev_close",
        "raw_open", "raw_high", "raw_low", "source_file"
    ]
].copy()

if original_match.duplicated(["Symbol", "Date"]).any():
    duplicate_rows = original_match[
        original_match.duplicated(["Symbol", "Date"], keep=False)
    ]
    duplicate_rows.to_csv(
        OUTPUT_DIR / "source_duplicate_keys.csv", index=False
    )
    raise ValueError(
        "Original CSVs contain duplicate Symbol-Date keys. "
        "See outputs/source_duplicate_keys.csv."
    )

matched = features.merge(
    original_match,
    on=["Symbol", "Date"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

matched["raw_close_match"] = (
    matched["_merge"].eq("both")
    & np.isclose(
        matched["Close Price"],
        matched["raw_close"],
        rtol=1e-7,
        atol=1e-6,
        equal_nan=False,
    )
)

matched["close_difference"] = (
    matched["Close Price"] - matched["raw_close"]
)

matched["source_match_status"] = np.select(
    [
        matched["_merge"].ne("both"),
        matched["raw_close_match"],
    ],
    [
        "NO_ORIGINAL_ROW_MATCH",
        "CLOSE_MATCH",
    ],
    default="CLOSE_MISMATCH",
)

comparison_cols = [
    "Symbol", "Date", "Close Price", "raw_close",
    "close_difference", "raw_prev_close", "source_file",
    "source_match_status",
]

comparison = matched[comparison_cols].copy()
comparison_path = OUTPUT_DIR / "original_vs_prepared_price_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\nOriginal vs prepared close comparison:")
print(comparison["source_match_status"].value_counts(dropna=False).to_string())


# ------------------------------------------------------------
# 3. FIND UNUSUAL MOVES IN ORIGINAL PRICES
# ------------------------------------------------------------

original = original.sort_values(["Symbol", "Date"]).reset_index(drop=True)
g = original.groupby("Symbol", sort=False)

original["previous_row_close"] = g["raw_close"].shift(1)
original["return_from_previous_row"] = (
    original["raw_close"] / original["previous_row_close"] - 1
)
original["return_from_reported_prev_close"] = (
    original["raw_close"] / original["raw_prev_close"] - 1
)
original["price_ratio_to_previous"] = (
    original["raw_close"] / original["previous_row_close"]
)

# A factor-like price change is a candidate only; not proof of an event.
common_factors = [2, 3, 4, 5, 10, 0.5, 1/3, 0.25, 0.2, 0.1]
factor_like = pd.Series(False, index=original.index)

for factor in common_factors:
    factor_like |= (
        np.isfinite(original["price_ratio_to_previous"])
        & (
            original["price_ratio_to_previous"] / factor - 1
        ).abs().le(FACTOR_TOLERANCE)
    )

large_change = (
    original["return_from_previous_row"].abs()
    >= LARGE_MOVE_THRESHOLD
).fillna(False)

large_prev_close_change = (
    original["return_from_reported_prev_close"].abs()
    >= LARGE_MOVE_THRESHOLD
).fillna(False)

candidate_mask = large_change | large_prev_close_change | factor_like

candidates = original.loc[candidate_mask].copy()

def reason_for_row(row):
    reasons = []
    r1 = row["return_from_previous_row"]
    r2 = row["return_from_reported_prev_close"]

    if pd.notna(r1) and abs(r1) >= LARGE_MOVE_THRESHOLD:
        reasons.append("Large close-to-close change")
    if pd.notna(r2) and abs(r2) >= LARGE_MOVE_THRESHOLD:
        reasons.append("Large change vs reported Prev Close")
    if bool(factor_like.loc[row.name]):
        reasons.append("Price ratio near common factor")
    return "; ".join(reasons)

candidates["review_reason"] = [
    reason_for_row(row) for _, row in candidates.iterrows()
]

candidate_cols = [
    "Symbol", "Date", "source_file", "raw_prev_close",
    "previous_row_close", "raw_open", "raw_high", "raw_low",
    "raw_close", "return_from_previous_row",
    "return_from_reported_prev_close", "price_ratio_to_previous",
    "review_reason",
]

candidate_path = OUTPUT_DIR / "corporate_action_candidates_source.csv"
candidates[candidate_cols].to_csv(candidate_path, index=False)

print(f"\nPrice-move candidate rows: {len(candidates)}")
if len(candidates):
    print(
        candidates[
            ["Symbol", "Date", "raw_prev_close", "raw_close",
             "return_from_previous_row", "review_reason"]
        ].head(25).to_string(index=False)
    )


# ------------------------------------------------------------
# 4. CREATE A CORPORATE-ACTION REGISTER TEMPLATE
# ------------------------------------------------------------

register_path = OUTPUT_DIR / "corporate_action_register_template.csv"

register_cols = [
    "Symbol",
    "Ex_Date",
    "Event_Type",
    "Ratio_or_Details",
    "Adjustment_Factor",
    "Official_Source",
    "Verified",
    "Notes",
]

if register_path.exists():
    print(
        "\nExisting register template found; it will NOT be overwritten:"
    )
else:
    register = candidates[["Symbol", "Date"]].drop_duplicates().copy()
    register = register.rename(columns={"Date": "Ex_Date"})

    for col in register_cols:
        if col not in register.columns:
            register[col] = ""

    register = register[register_cols].sort_values(["Symbol", "Ex_Date"])
    register.to_csv(register_path, index=False)

    print(f"\nCreated register template: {register_path}")


# ------------------------------------------------------------
# 5. REPORT DATASET COVERAGE AND SUMMARY
# ------------------------------------------------------------

coverage = {
    "original_rows": int(len(original)),
    "original_symbols": int(original["Symbol"].nunique()),
    "prepared_rows": int(len(features)),
    "prepared_symbols": int(features["Symbol"].nunique()),
    "prepared_rows_with_original_match": int(
        matched["_merge"].eq("both").sum()
    ),
    "prepared_rows_without_original_match": int(
        matched["_merge"].ne("both").sum()
    ),
    "close_matches": int(
        matched["source_match_status"].eq("CLOSE_MATCH").sum()
    ),
    "close_mismatches": int(
        matched["source_match_status"].eq("CLOSE_MISMATCH").sum()
    ),
    "candidate_price_move_rows": int(len(candidates)),
    "corporate_action_adjustment_status": "UNVERIFIED",
    "note": (
        "Large price moves are candidates only. Confirm events using "
        "official NSE corporate-action records. No price adjustments "
        "or source data modifications were performed."
    ),
}

summary_path = OUTPUT_DIR / "corporate_action_source_audit_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(coverage, f, indent=2, default=str)

print("\n" + "=" * 72)
print("AUDIT COMPLETE")
print("=" * 72)
print(f"File inventory:         {inventory_path}")
print(f"Price comparison:       {comparison_path}")
print(f"Candidate events:       {candidate_path}")
print(f"Register template:      {register_path}")
print(f"Summary:                {summary_path}")
print("\nNo original or prepared dataset was modified.")
