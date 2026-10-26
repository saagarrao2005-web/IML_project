import pandas as pd
from pathlib import Path

# ===== CONFIGURATION =====
INPUT_FILE = Path("Datasets/25-10-2021-TO-25-10-2026-YESBANK-ALL-N.csv")
OUTPUT_FILE = Path("cleaned_dataset_25-10-2021-TO-25-10-2026-YESBANK-ALL-N.csv")
REPORT_FILE = Path("missing_values_report.csv")

DATE_COL = "Date"
SYMBOL_COL = "Symbol"

# Only list columns where carrying the last known value forward
# is justified by the meaning of the data.
FILL_COLUMNS = [
    # "Fundamental_Ratio",
    # "Market_Cap",
]

# ===== LOAD CSV =====
df = pd.read_csv(
    INPUT_FILE,
    na_values=["-", "", "N/A", "NA", "null", "None"],
    keep_default_na=True
)

# Clean column headers
df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)

# Parse dates and sort chronologically
if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL], errors="coerce", dayfirst=True
    )
    df = df.sort_values(
        [SYMBOL_COL, DATE_COL] if SYMBOL_COL in df.columns
        else [DATE_COL]
    ).reset_index(drop=True)

# ===== REPORT BEFORE CLEANING =====
before = pd.DataFrame({
    "missing_before": df.isna().sum(),
    "percentage_before": (df.isna().mean() * 100).round(2)
})

print("\nMissing values before cleaning:")
print(before[before["missing_before"] > 0])

# ===== FILL APPROVED COLUMNS ONLY =====
for col in FILL_COLUMNS:
    if col not in df.columns:
        print(f"Warning: column '{col}' not found.")
        continue

    if SYMBOL_COL in df.columns:
        df[col] = df.groupby(
            SYMBOL_COL, sort=False
        )[col].ffill()
    else:
        df[col] = df[col].ffill()

# ===== REPORT AFTER CLEANING =====
after = pd.DataFrame({
    "missing_after": df.isna().sum(),
    "percentage_after": (df.isna().mean() * 100).round(2)
})

report = before.join(after)
report.to_csv(REPORT_FILE, index=True)

# ===== SAVE CLEANED CSV =====
df.to_csv(OUTPUT_FILE, index=False)

print("\nMissing values after cleaning:")
print(after[after["missing_after"] > 0])

print("\nFiles created:")
print("Cleaned dataset:", OUTPUT_FILE)
print("Missing-value report:", REPORT_FILE)
print("\nFinal dataset shape:", df.shape)
print("\nNote: Unapproved columns and remaining NaNs were not filled.")
