
from pathlib import Path
import json
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(
    r"D:\B.Tech\Sem3\IML\Project\Stock_Prediction"
)

DATA_DIR = BASE_DIR / "cleaned_Dataset"
OUTPUT_DIR = BASE_DIR / "model_development" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5  # Predict direction 5 trading sessions ahead

# --------------------------------------------------
# 2. LOAD ALL STOCK FILES
# --------------------------------------------------

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

frames = []

for file in files:
    df = pd.read_csv(file)

    required = [
        "Symbol", "Date", "Close Price",
        "Open Price", "High Price", "Low Price",
        "Total Traded Quantity"
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{file.name} is missing columns: {sorted(missing)}"
        )

    # Normalize dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert Indian-formatted numbers to numeric values
    numeric_cols = [
        "Prev Close", "Open Price", "High Price",
        "Low Price", "Last Price", "Close Price",
        "Average Price", "Total Traded Quantity",
        "Turnover ₹", "No. of Trades", "Deliverable Qty",
        "% Dly Qt to Traded Qty"
    ]

    for col in numeric_cols:
        if col in df.columns:
            values = (
                df[col].astype("string")
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(values, errors="coerce")

    # Keep the symbol supplied inside the CSV
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print("Files loaded:", len(files))
print("Combined rows before checks:", len(data))
print("Unique stocks:", data["Symbol"].nunique())

# --------------------------------------------------
# 3. BASIC DATA QUALITY CHECKS
# --------------------------------------------------

invalid_dates = int(data["Date"].isna().sum())

data = data.dropna(subset=["Date", "Symbol", "Close Price"]).copy()

# Keep valid NSE equity-series rows
if "Series" in data.columns:
    data["Series"] = data["Series"].astype(str).str.strip()
    data = data[data["Series"] == "EQ"].copy()

# Count duplicate stock-date keys
duplicate_rows = int(
    data.duplicated(["Symbol", "Date"], keep=False).sum()
)

# Stop instead of silently selecting arbitrary duplicate rows
if duplicate_rows:
    raise ValueError(
        f"Found {duplicate_rows} rows involved in duplicate "
        "Symbol-Date keys. Resolve these before training."
    )

# Sort chronologically within each stock
data = data.sort_values(["Symbol", "Date"]).reset_index(drop=True)

# Price sanity checks
bad_prices = int((data["Close Price"] <= 0).sum())

if bad_prices:
    raise ValueError(f"Found {bad_prices} non-positive closing prices.")

# Check OHLC consistency
bad_ohlc = (
    (data["High Price"] < data["Low Price"])
    | (data["High Price"] < data["Open Price"])
    | (data["High Price"] < data["Close Price"])
    | (data["Low Price"] > data["Open Price"])
    | (data["Low Price"] > data["Close Price"])
)

bad_ohlc_rows = int(bad_ohlc.fillna(False).sum())

print("Invalid dates dropped:", invalid_dates)
print("Duplicate stock-date rows:", duplicate_rows)
print("Non-positive closing prices:", bad_prices)
print("OHLC inconsistency rows:", bad_ohlc_rows)

if bad_ohlc_rows:
    raise ValueError(
        "OHLC consistency check failed. Investigate these rows "
        "before training."
    )

# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

grouped = data.groupby("Symbol", sort=False)
close = grouped["Close Price"]

# Historical returns
for lag in [1, 2, 5, 10, 20]:
    data[f"return_{lag}d"] = (
        close.pct_change(lag, fill_method=None)
    )

# Moving-average ratios
for window in [5, 10, 20, 50]:
    moving_average = close.transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )

    data[f"close_vs_sma_{window}"] = (
        data["Close Price"] / moving_average - 1
    )

# Historical volatility: rolling standard deviation of daily returns
daily_return = grouped["Close Price"].pct_change(fill_method=None)

data["volatility_10d"] = daily_return.groupby(
    data["Symbol"]
).transform(
    lambda s: s.rolling(10, min_periods=10).std()
)

data["volatility_20d"] = daily_return.groupby(
    data["Symbol"]
).transform(
    lambda s: s.rolling(20, min_periods=20).std()
)

# Daily price range and close location
data["daily_range_pct"] = (
    data["High Price"] - data["Low Price"]
) / data["Close Price"]

data["open_close_return"] = (
    data["Close Price"] / data["Open Price"] - 1
)

# Volume and trading activity
if "Total Traded Quantity" in data.columns:
    data["volume_change_1d"] = grouped[
        "Total Traded Quantity"
    ].pct_change(fill_method=None)

    volume_avg = grouped["Total Traded Quantity"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )

    data["volume_vs_avg20"] = (
        data["Total Traded Quantity"] / volume_avg - 1
    )

if "No. of Trades" in data.columns:
    trades_avg = grouped["No. of Trades"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )

    data["trades_vs_avg20"] = data["No. of Trades"] / trades_avg - 1

# Calendar information
data["day_of_week"] = data["Date"].dt.dayofweek
data["month"] = data["Date"].dt.month

# --------------------------------------------------
# 5. CREATE THE FUTURE TARGET
# --------------------------------------------------

future_close = close.shift(-HORIZON)

data["future_return_5d"] = (
    future_close / data["Close Price"] - 1
)

# Binary target: 1 = higher closing price after 5 sessions

data["target_up_5d"] = (
    (future_close > data["Close Price"])
    .astype("float64")
)

# The final 5 rows per stock have no known future price.
data.loc[future_close.isna(), "target_up_5d"] = np.nan


# --------------------------------------------------
# 6. FINAL DATASET AND REPORT
# --------------------------------------------------

# These columns are not predictors.
# Future return is retained for evaluation only.
metadata_cols = ["Symbol", "Date", "Series"]

feature_cols = [
    col for col in data.columns
    if col not in metadata_cols
    and col not in ["future_return_5d", "target_up_5d"]
]

# Save engineered data, including the label and evaluation return.
# Missing rolling features at the beginning of each stock's history
# are expected. Missing delivery data is not automatically filled.
output_file = OUTPUT_DIR / "stock_features.csv"

data.to_csv(output_file, index=False)

# Rows eligible for model training must have a target.
model_rows = data.dropna(subset=["target_up_5d"]).copy()

report = {
    "input_files": len(files),
    "stocks": int(data["Symbol"].nunique()),
    "rows_after_quality_checks": int(len(data)),
    "rows_with_target": int(len(model_rows)),
    "date_start": str(data["Date"].min().date()),
    "date_end": str(data["Date"].max().date()),
    "duplicate_stock_date_rows": duplicate_rows,
    "invalid_dates_dropped": invalid_dates,
    "ohlc_inconsistency_rows": bad_ohlc_rows,
    "prediction_horizon_sessions": HORIZON,
    "target_definition": "1 if close at t+5 > close at t, else 0",
    "target_class_counts": {
        str(int(k)): int(v)
        for k, v in model_rows["target_up_5d"]
        .value_counts().sort_index().items()
    },
    "missing_values": {
        k: int(v)
        for k, v in data.isna().sum().items()
        if v > 0
    },
    "feature_columns": feature_cols,
}

report_file = OUTPUT_DIR / "preparation_report.json"

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 55)
print("FEATURE PREPARATION COMPLETED")
print("=" * 55)
print("Output dataset:", output_file)
print("Quality report:", report_file)
print("Total rows:", len(data))
print("Rows with target:", len(model_rows))
print("Stocks:", data["Symbol"].nunique())
print("\nTarget distribution:")
print(model_rows["target_up_5d"].value_counts().sort_index())
print("\nMissing values:")
print(data.isna().sum()[data.isna().sum() > 0])
