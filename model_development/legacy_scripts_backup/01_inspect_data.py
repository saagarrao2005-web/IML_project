from pathlib import Path
import pandas as pd

# Your existing cleaned dataset folder
DATA_DIR = Path(
    r"D:\B.Tech\Sem3\IML\Project\Stock_Prediction\cleaned_Dataset"
)

# Find all CSV files
files = sorted(DATA_DIR.glob("*.csv"))

print("Number of CSV files found:", len(files))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}"
    )

all_data = []

for file in files:
    df = pd.read_csv(file)

    # Infer stock symbol from the filename
    parts = file.stem.split("-")
    symbol = parts[-3] if len(parts) >= 3 else "UNKNOWN"

    print("\n" + "=" * 65)
    print("File:", file.name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    df["source_symbol"] = symbol
    all_data.append(df)

print("\nInspection completed.")
print("Total files inspected:", len(all_data))
