
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR.parent / "cleaned_dataset"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

frames = []

for path in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    date_col = next(
        (c for c in df.columns if c.lower() in
         ["date", "trade date", "timestamp"]),
        None,
    )
    close_col = next(
        (c for c in df.columns if c.lower() in
         ["close price", "close", "closing price"]),
        None,
    )
    symbol_col = next(
        (c for c in df.columns if c.lower() in
         ["symbol", "ticker"]),
        None,
    )
    series_col = next(
        (c for c in df.columns if c.lower() == "series"),
        None,
    )

    if date_col is None or close_col is None:
        print(f"Skipping {path.name}: date/close column not found")
        continue

    df["Date_parsed"] = pd.to_datetime(
        df[date_col], errors="coerce", dayfirst=True
    )
    df["Close_parsed"] = pd.to_numeric(
        df[close_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    if symbol_col:
        df["Symbol_parsed"] = (
            df[symbol_col].astype(str).str.strip().str.upper()
        )
    else:
        df["Symbol_parsed"] = path.stem.strip().upper()

    df["source_file"] = path.name
    df["source_row_number"] = range(2, len(df) + 2)

    if series_col:
        df["Series_parsed"] = df[series_col].astype(str).str.strip()
    else:
        df["Series_parsed"] = ""

    frames.append(df)

if not frames:
    raise ValueError(
        f"No readable CSVs with recognized Date and Close columns in {RAW_DIR}"
    )

all_rows = pd.concat(frames, ignore_index=True)

# Invalid dates cannot form a reliable Symbol-Date key.
invalid_dates = all_rows[all_rows["Date_parsed"].isna()].copy()
invalid_dates.to_csv(OUT_DIR / "invalid_date_rows.csv", index=False)

valid_rows = all_rows[all_rows["Date_parsed"].notna()].copy()
key = ["Symbol_parsed", "Date_parsed"]

duplicates = valid_rows[
    valid_rows.duplicated(key, keep=False)
].copy()

if duplicates.empty:
    print("No duplicate Symbol-Date rows found.")
else:
    summary = (
        duplicates.groupby(key, dropna=False)
        .agg(
            duplicate_rows=("source_file", "size"),
            files=(
                "source_file",
                lambda s: " | ".join(sorted(set(s))),
            ),
            close_values=(
                "Close_parsed",
                lambda s: " | ".join(map(str, s)),
            ),
            series_values=(
                "Series_parsed",
                lambda s: " | ".join(map(str, s)),
            ),
        )
        .reset_index()
    )

    detail_path = OUT_DIR / "duplicate_date_diagnostic_details.csv"
    summary_path = OUT_DIR / "duplicate_date_diagnostic_summary.csv"

    duplicates.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Duplicate rows: {len(duplicates)}")
    print(f"Duplicate Symbol-Date groups: {len(summary)}")
    print("\nFirst 30 duplicate groups:")
    print(summary.head(30).to_string(index=False))
    print(f"\nDetails saved to: {detail_path}")
    print(f"Summary saved to: {summary_path}")

print(f"\nTotal input rows: {len(all_rows)}")
print(f"Rows with invalid dates: {len(invalid_dates)}")
print(f"Rows with valid dates: {len(valid_rows)}")
print("\nDiagnostic complete. No source data was modified.")
