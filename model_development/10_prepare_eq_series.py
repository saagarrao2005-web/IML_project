
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR.parent / "cleaned_dataset"
OUT_DIR = BASE_DIR / "outputs" / "equity_series_only"
REPORT_DIR = BASE_DIR / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

all_eq = []
report = []

for path in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    series_col = next(
        (c for c in df.columns if c.lower() == "series"), None
    )
    symbol_col = next(
        (c for c in df.columns if c.lower() in ["symbol", "ticker"]),
        None,
    )
    date_col = next(
        (c for c in df.columns if c.lower() in
         ["date", "trade date", "timestamp"]),
        None,
    )

    if series_col is None or date_col is None:
        raise ValueError(
            f"{path.name}: Series or Date column missing. "
            "No files were intentionally filtered without these fields."
        )

    original_count = len(df)

    series = df[series_col].astype(str).str.strip().str.upper()
    eq = df.loc[series == "EQ"].copy()

    if symbol_col:
        eq["_audit_symbol"] = (
            eq[symbol_col].astype(str).str.strip().str.upper()
        )
    else:
        eq["_audit_symbol"] = path.stem.strip().upper()

    eq["_audit_date"] = pd.to_datetime(
        eq[date_col], errors="coerce"
    ).dt.normalize()

    invalid_dates = int(eq["_audit_date"].isna().sum())
    valid = eq.loc[eq["_audit_date"].notna()].copy()

    duplicate_mask = valid.duplicated(
        ["_audit_symbol", "_audit_date"], keep=False
    )
    duplicate_rows = int(duplicate_mask.sum())

    # Keep audit-only columns out of the prepared CSV.
    valid = valid.drop(columns=["_audit_symbol", "_audit_date"])

    # Preserve the original columns and values; no rows are silently
    # removed for duplicate keys.
    destination = OUT_DIR / path.name
    valid.to_csv(destination, index=False)

    report.append({
        "file": path.name,
        "original_rows": original_count,
        "EQ_rows": int((series == "EQ").sum()),
        "EQ_valid_date_rows": len(valid),
        "EQ_invalid_dates": invalid_dates,
        "EQ_duplicate_key_rows": duplicate_rows,
        "non_EQ_rows_excluded": int((series != "EQ").sum()),
        "output_file": str(destination),
    })

    if not valid.empty:
        temp = valid.copy()
        temp["audit_symbol"] = (
            temp[symbol_col].astype(str).str.strip().str.upper()
            if symbol_col else path.stem.strip().upper()
        )
        temp["audit_date"] = pd.to_datetime(
            temp[date_col], errors="coerce"
        ).dt.normalize()
        all_eq.append(temp)

report_df = pd.DataFrame(report)
report_path = REPORT_DIR / "eq_series_preparation_report.csv"
report_df.to_csv(report_path, index=False)

print("=" * 65)
print("EQ SERIES PREPARATION REPORT")
print("=" * 65)
print(report_df.to_string(index=False))

if all_eq:
    combined = pd.concat(all_eq, ignore_index=True)
    dup = combined[
        combined.duplicated(
            ["audit_symbol", "audit_date"], keep=False
        )
    ]
    dup.to_csv(
        REPORT_DIR / "eq_series_duplicate_keys.csv", index=False
    )

    print("\nTotal EQ rows:", len(combined))
    print("Remaining duplicate-key rows:", len(dup))
    print("Unique symbols:", combined["audit_symbol"].nunique())
    print("Preparation report:", report_path)
    print("EQ-only audit copies:", OUT_DIR)

    if len(dup):
        print(
            "\nWARNING: EQ series still contains duplicate "
            "Symbol-Date keys. Investigate before proceeding."
        )
    else:
        print("\nPASS: EQ Symbol-Date keys are unique.")
else:
    print("ERROR: No EQ rows found. Do not continue.")
