"""
Data Pipeline Module for the NSE Stock Price Prediction ML Project.

Handles:
1. Robust ingestion and normalization of NSE equity data.
2. Rigorous data quality and schema auditing.
3. Corporate action backward-adjustment (splits, bonus issues, demerger).
4. Leakage-free feature engineering based exclusively on information
   available at the market close of trading observation t.
5. Per-symbol 5-trading-session forward target construction and alignment auditing.
6. Persistence of clean analytical datasets and audit artifacts.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    RAW_DATA_DIR,
    EQ_DATA_DIR,
    DATASETS_DIR,
    REPORTS_DIR,
    ANALYTICAL_DATASET_PATH,
    CORPORATE_ACTION_REGISTER_PATH,
    DATA_QUALITY_REPORT_PATH,
    TARGET_ALIGNMENT_REPORT_PATH,
    FEATURE_DEFINITIONS_PATH,
    HORIZON_SESSIONS,
    HORIZON_REGRESSION_SESSIONS,
    EXPECTED_SYMBOLS,
    CONFIRMED_CORPORATE_ACTIONS,
    VERIFIED_MARKET_MOVES,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
)


def load_raw_data(data_dir: Path = None) -> pd.DataFrame:
    """
    Load daily stock data across all 16 symbols.
    Prioritizes the verified EQ-only series directory.
    Falls back to raw directory with Series == 'EQ' filter if needed.
    """
    if data_dir is None:
        data_dir = EQ_DATA_DIR if EQ_DATA_DIR.exists() and list(EQ_DATA_DIR.glob("*.csv")) else RAW_DATA_DIR

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

    frames = []
    for file_path in csv_files:
        df = pd.read_csv(file_path, low_memory=False)
        # Strip potential BOM and whitespace from headers
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

        # Case-insensitive column resolution
        col_map = {c.lower(): c for c in df.columns}
        symbol_col = col_map.get("symbol") or col_map.get("ticker")
        date_col = col_map.get("date") or col_map.get("trade date") or col_map.get("timestamp")
        series_col = col_map.get("series")

        if not symbol_col or not date_col:
            raise ValueError(f"{file_path.name} is missing required Symbol or Date column.")

        # Standardize core column names
        rename_dict = {
            symbol_col: "Symbol",
            date_col: "Date",
        }
        if series_col:
            rename_dict[series_col] = "Series"
        df = df.rename(columns=rename_dict)

        # Standardize symbol and filter series
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        if "Series" in df.columns:
            df["Series"] = df["Series"].astype(str).str.strip().str.upper()
            df = df[df["Series"] == "EQ"].copy()
        else:
            df["Series"] = "EQ"

        # Standardize Date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

        # Parse numeric columns cleanly
        numeric_targets = [
            "Prev Close", "Open Price", "High Price", "Low Price",
            "Last Price", "Close Price", "Average Price",
            "Total Traded Quantity", "Turnover ₹", "No. of Trades",
            "Deliverable Qty", "% Dly Qt to Traded Qty"
        ]
        for col in numeric_targets:
            actual_col = col_map.get(col.lower())
            if actual_col and actual_col in df.columns:
                cleaned_str = (
                    df[actual_col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("₹", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(
                    cleaned_str.replace({"": np.nan, "-": np.nan, "--": np.nan, "NA": np.nan, "null": np.nan}),
                    errors="coerce"
                )

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return data


def run_data_quality_audit(df: pd.DataFrame) -> dict:
    """
    Perform rigorous validation on dates, duplicate keys, OHLC consistency,
    and volume/trade counts.
    """
    audit = {}
    audit["total_rows"] = int(len(df))
    audit["unique_symbols"] = int(df["Symbol"].nunique())
    audit["symbols_list"] = sorted(df["Symbol"].unique().tolist())
    audit["date_min"] = str(df["Date"].min().date())
    audit["date_max"] = str(df["Date"].max().date())

    # Invalid dates check
    invalid_dates = int(df["Date"].isna().sum())
    audit["invalid_dates"] = invalid_dates

    # Duplicate Symbol-Date keys check
    duplicate_keys = int(df.duplicated(subset=["Symbol", "Date"], keep=False).sum())
    audit["duplicate_symbol_date_keys"] = duplicate_keys

    # Price non-negativity check
    non_positive_close = int((df["Close Price"] <= 0).sum())
    audit["non_positive_close_prices"] = non_positive_close

    # OHLC internal consistency
    ohlc_violations = (
        (df["High Price"] < df["Low Price"])
        | (df["High Price"] < df["Open Price"])
        | (df["High Price"] < df["Close Price"])
        | (df["Low Price"] > df["Open Price"])
        | (df["Low Price"] > df["Close Price"])
    )
    audit["ohlc_inconsistencies"] = int(ohlc_violations.sum())

    # Trading activity bounds
    audit["zero_or_negative_volume"] = int((df["Total Traded Quantity"] <= 0).sum())
    audit["zero_or_negative_trades"] = int((df["No. of Trades"] <= 0).sum())
    audit["zero_or_negative_deliverable"] = int((df["Deliverable Qty"] <= 0).sum())

    # Save report
    with open(DATA_QUALITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # Assert critical data integrity invariants
    if invalid_dates > 0:
        raise ValueError(f"Data Quality Failure: Found {invalid_dates} rows with invalid dates.")
    if duplicate_keys > 0:
        raise ValueError(f"Data Quality Failure: Found {duplicate_keys} duplicate Symbol-Date rows.")
    if non_positive_close > 0:
        raise ValueError(f"Data Quality Failure: Found {non_positive_close} non-positive close prices.")
    if audit["ohlc_inconsistencies"] > 0:
        raise ValueError(f"Data Quality Failure: Found {audit['ohlc_inconsistencies']} OHLC inconsistency rows.")

    return audit


def apply_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies backward price and volume adjustments for confirmed corporate actions.

    Convention:
    - Current prices after the event remain in raw Rupee terms (adjustment factor = 1.0).
    - Historical prices before the ex-date are multiplied by the cumulative adjustment factor:
      Adj_Price(t) = Raw_Price(t) * factor
    - Historical volumes before the ex-date are multiplied by (1.0 / factor):
      Adj_Volume(t) = Raw_Volume(t) / factor
    This ensures that:
    1. Returns across the ex-date are economically accurate and free from artificial drops.
    2. Total turnover (Price * Volume) remains invariant.
    3. Raw prices are preserved side-by-side for complete traceability.
    """
    data = df.copy()

    # Retain raw unadjusted series
    data["raw_close"] = data["Close Price"]
    data["raw_open"] = data["Open Price"]
    data["raw_high"] = data["High Price"]
    data["raw_low"] = data["Low Price"]
    data["raw_prev_close"] = data["Prev Close"]
    data["raw_volume"] = data["Total Traded Quantity"]

    # Initialize adjustment factor column
    data["adj_factor"] = 1.0

    register_rows = []

    for symbol, group in data.groupby("Symbol"):
        if symbol in CONFIRMED_CORPORATE_ACTIONS:
            info = CONFIRMED_CORPORATE_ACTIONS[symbol]
            ex_date = pd.to_datetime(info["ex_date"])
            factor = float(info["adjustment_factor"])

            # Mask for historical observations strictly prior to ex-date
            hist_mask = (data["Symbol"] == symbol) & (data["Date"] < ex_date)
            affected_rows = int(hist_mask.sum())
            data.loc[hist_mask, "adj_factor"] *= factor

            register_rows.append({
                "symbol": symbol,
                "event_date": info["ex_date"],
                "event_type": info["action_type"],
                "ratio_details": info["ratio_details"],
                "adjustment_factor": factor,
                "verification_source": info["source"],
                "verification_status": info["status"],
                "affected_observations": affected_rows,
                "notes": info["notes"],
            })

    # Add verified non-corporate-action market moves to register for transparency
    for move in VERIFIED_MARKET_MOVES:
        register_rows.append({
            "symbol": move["symbol"],
            "event_date": ", ".join(move["dates"]),
            "event_type": "Market Price Movement (Non-Action)",
            "ratio_details": "N/A - Genuine supply/demand price discovery",
            "adjustment_factor": 1.0,
            "verification_source": "Exchange trade logs, news releases, earnings reports",
            "verification_status": "VERIFIED_MARKET_MOVE",
            "affected_observations": 0,
            "notes": move["reason"],
        })

    register_df = pd.DataFrame(register_rows)
    register_df.to_csv(CORPORATE_ACTION_REGISTER_PATH, index=False)

    # Compute adjusted price and volume series
    data["adj_close"] = data["raw_close"] * data["adj_factor"]
    data["adj_open"] = data["raw_open"] * data["adj_factor"]
    data["adj_high"] = data["raw_high"] * data["adj_factor"]
    data["adj_low"] = data["raw_low"] * data["adj_factor"]
    data["adj_volume"] = data["raw_volume"] / data["adj_factor"]

    return data


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all predictive features strictly per stock.
    Guarantees no future information or target data enters the features.
    """
    data = df.copy()
    grouped = data.groupby("Symbol", sort=False)

    # 1. Multi-horizon returns on adjusted close
    for lag in [1, 2, 5, 10, 20]:
        data[f"return_{lag}d"] = grouped["adj_close"].pct_change(lag)

    # 2. Moving average distance ratios
    for window in [5, 10, 20, 50]:
        sma = grouped["adj_close"].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        data[f"close_vs_sma_{window}"] = data["adj_close"] / sma - 1.0

    # 3. Rolling volatility (standard deviation of daily return)
    daily_ret = grouped["adj_close"].pct_change()
    data["volatility_10d"] = daily_ret.groupby(data["Symbol"]).transform(
        lambda s: s.rolling(10, min_periods=10).std()
    )
    data["volatility_20d"] = daily_ret.groupby(data["Symbol"]).transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )

    # 4. Intraday price structure
    data["daily_range_pct"] = (data["adj_high"] - data["adj_low"]) / data["adj_close"]
    data["open_close_return"] = data["adj_close"] / data["adj_open"] - 1.0

    # 5. Volume and trading activity
    data["volume_change_1d"] = grouped["adj_volume"].pct_change()

    vol_avg20 = grouped["adj_volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    # Guard against division by zero or NaN producing inf values
    data["volume_vs_avg20"] = np.where(
        (vol_avg20 > 0) & vol_avg20.notna(),
        data["adj_volume"] / vol_avg20 - 1.0,
        np.nan,
    )

    trades_avg20 = grouped["No. of Trades"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    data["trades_vs_avg20"] = np.where(
        (trades_avg20 > 0) & trades_avg20.notna(),
        data["No. of Trades"] / trades_avg20 - 1.0,
        np.nan,
    )

    # Delivery percentage
    if "% Dly Qt to Traded Qty" in data.columns:
        data["delivery_pct"] = data["% Dly Qt to Traded Qty"]
    else:
        data["delivery_pct"] = (data["Deliverable Qty"] / data["Total Traded Quantity"]) * 100.0

    # 6. Technical Indicators: RSI 14 (Wilder's formulation)
    def compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
        delta = close_series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Leave warmup NaN for the imputer to handle during training;
        # filling with 50.0 here would introduce look-ahead bias by injecting
        # a "neutral" value before the indicator has enough data to compute.
        return rsi

    data["rsi_14"] = grouped["adj_close"].transform(compute_rsi)

    # 7. Calendar information
    data["day_of_week"] = data["Date"].dt.dayofweek
    data["month"] = data["Date"].dt.month

    return data


def construct_target(
    df: pd.DataFrame,
    horizon: int = HORIZON_SESSIONS,
    horizon_reg: int = HORIZON_REGRESSION_SESSIONS
) -> pd.DataFrame:
    """
    Constructs forward prediction targets per stock.
    
    Task A: 5-Trading-Session Directional Classification
    - Uses properly adjusted future close 5 trading sessions ahead (t + 5).
    - future_return_5d = adj_close(t+5) / adj_close(t) - 1.0
    - target_up_5d = 1.0 if future_return_5d > 0.0 else 0.0
    - Ties (future close exactly equal to current close) are assigned 0.0.
    - Final horizon sessions per stock have no known future price and are set to NaN.

    Task B: 3-Month (63 Trading Sessions) Continuous-Return Regression
    - Uses properly adjusted future close 63 trading sessions ahead (t + 63).
    - future_return_63d = adj_close(t+63) / adj_close(t) - 1.0
    - Final 63 sessions per stock have no known future price and are set to NaN.
    """
    data = df.copy()
    grouped = data.groupby("Symbol", sort=False)

    # Task A: 5-session forward return and classification target
    future_adj_close_5d = grouped["adj_close"].shift(-horizon)
    data["future_adj_close_5d"] = future_adj_close_5d
    data["future_return_5d"] = future_adj_close_5d / data["adj_close"] - 1.0
    has_future_5d = future_adj_close_5d.notna()
    data["target_up_5d"] = np.where(
        has_future_5d,
        (future_adj_close_5d > data["adj_close"]).astype(float),
        np.nan
    )

    # Task B: 63-session forward return target
    future_adj_close_63d = grouped["adj_close"].shift(-horizon_reg)
    data["future_adj_close_63d"] = future_adj_close_63d
    data["future_return_63d"] = future_adj_close_63d / data["adj_close"] - 1.0

    return data


def audit_target_alignment(df: pd.DataFrame, horizon: int = HORIZON_SESSIONS) -> dict:
    """
    Verifies that the target horizon strictly shifts within each symbol
    and never leaks across symbol boundaries or uses unadjusted discontinuities.
    """
    grouped = df.groupby("Symbol", sort=False)
    total_rows = len(df)
    unique_symbols = df["Symbol"].nunique()
    expected_nan_targets = unique_symbols * horizon

    actual_nan_targets = int(df["target_up_5d"].isna().sum())

    # Verify per-symbol alignment for Task A
    symbol_shifts_ok = True
    for symbol, group in grouped:
        group_close = group["adj_close"].to_numpy()
        group_target = group["target_up_5d"].to_numpy()

        for i in range(len(group_close) - horizon):
            curr_c = group_close[i]
            fwd_c = group_close[i + horizon]
            expected_lbl = 1.0 if fwd_c > curr_c else 0.0
            actual_lbl = group_target[i]
            if actual_lbl != expected_lbl:
                symbol_shifts_ok = False
                break

    # Verify per-symbol alignment for Task B (63-day regression)
    expected_nan_reg = unique_symbols * HORIZON_REGRESSION_SESSIONS
    actual_nan_reg = int(df["future_return_63d"].isna().sum())
    reg_shifts_ok = True
    for symbol, group in grouped:
        group_close = group["adj_close"].to_numpy()
        group_ret = group["future_return_63d"].to_numpy()
        for i in range(len(group_close) - HORIZON_REGRESSION_SESSIONS):
            expected_ret = (group_close[i + HORIZON_REGRESSION_SESSIONS] / group_close[i]) - 1.0
            if not np.isclose(group_ret[i], expected_ret, rtol=1e-10):
                reg_shifts_ok = False
                break

    # Check ties
    ties_count = int(
        (df["target_up_5d"].notna() & (df["future_adj_close_5d"] == df["adj_close"])).sum()
    )

    report = {
        "total_rows": total_rows,
        "unique_symbols": unique_symbols,
        "task_a_5d": {
            "expected_nan_targets": expected_nan_targets,
            "actual_nan_targets": actual_nan_targets,
            "nan_count_exact_match": (actual_nan_targets == expected_nan_targets),
            "target_per_symbol_alignment_verified": symbol_shifts_ok,
            "ties_count": ties_count,
            "class_1_count": int((df["target_up_5d"] == 1.0).sum()),
            "class_0_count": int((df["target_up_5d"] == 0.0).sum()),
            "class_1_ratio": float((df["target_up_5d"] == 1.0).mean()),
        },
        "task_b_63d": {
            "expected_nan_targets": expected_nan_reg,
            "actual_nan_targets": actual_nan_reg,
            "nan_count_exact_match": (actual_nan_reg == expected_nan_reg),
            "return_alignment_verified": reg_shifts_ok,
            "mean_forward_return": float(df["future_return_63d"].mean()),
            "std_forward_return": float(df["future_return_63d"].std()),
        },
        # Backwards compatibility keys
        "expected_nan_targets": expected_nan_targets,
        "actual_nan_targets": actual_nan_targets,
        "nan_count_exact_match": (actual_nan_targets == expected_nan_targets),
        "target_per_symbol_alignment_verified": symbol_shifts_ok,
        "ties_count": ties_count,
        "class_1_count": int((df["target_up_5d"] == 1.0).sum()),
        "class_0_count": int((df["target_up_5d"] == 0.0).sum()),
        "class_1_ratio": float((df["target_up_5d"] == 1.0).mean()),
    }

    with open(TARGET_ALIGNMENT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not symbol_shifts_ok:
        raise ValueError("Target Alignment Audit Failed: Mismatch between shifted prices and target label.")
    if actual_nan_targets != expected_nan_targets:
        raise ValueError(
            f"Target Alignment Audit Failed: Expected {expected_nan_targets} NaN targets, got {actual_nan_targets}."
        )

    return report


def save_feature_definitions():
    """Generates the Markdown documentation for all engineered features."""
    content = """# Feature Definitions and Availability Report

This document records the exact mathematical formula, warmup window, missing-value behavior, and information availability timestamp for every predictive feature used in the NSE Stock Price Prediction ML system.

## 1. Information Availability Guarantee
All features are constructed strictly using market data published on or before the close of trading observation $t$. No future prices, targets, or overlapping returns are utilized in feature creation. Preprocessing (imputation and scaling) is fitted exclusively on training folds during cross-validation and evaluation to prevent look-ahead contamination.

## 2. Feature Catalog

| Feature Name | Category | Mathematical Definition | Warmup Period | Missing Value Handling |
| :--- | :--- | :--- | :--- | :--- |
| `return_1d` | Momentum | $P_{t} / P_{t-1} - 1$ | 1 session | Median imputed in training fold |
| `return_2d` | Momentum | $P_{t} / P_{t-2} - 1$ | 2 sessions | Median imputed in training fold |
| `return_5d` | Momentum | $P_{t} / P_{t-5} - 1$ | 5 sessions | Median imputed in training fold |
| `return_10d` | Momentum | $P_{t} / P_{t-10} - 1$ | 10 sessions | Median imputed in training fold |
| `return_20d` | Momentum | $P_{t} / P_{t-20} - 1$ | 20 sessions | Median imputed in training fold |
| `close_vs_sma_5` | Trend Ratio | $P_{t} / \text{SMA}_5(P) - 1$ | 5 sessions | Median imputed in training fold |
| `close_vs_sma_10` | Trend Ratio | $P_{t} / \text{SMA}_{10}(P) - 1$ | 10 sessions | Median imputed in training fold |
| `close_vs_sma_20` | Trend Ratio | $P_{t} / \text{SMA}_{20}(P) - 1$ | 20 sessions | Median imputed in training fold |
| `close_vs_sma_50` | Trend Ratio | $P_{t} / \text{SMA}_{50}(P) - 1$ | 50 sessions | Median imputed in training fold |
| `volatility_10d` | Risk | $\text{Std}(\text{return}_{1d}, 10)$ | 10 sessions | Median imputed in training fold |
| `volatility_20d` | Risk | $\text{Std}(\text{return}_{1d}, 20)$ | 20 sessions | Median imputed in training fold |
| `daily_range_pct` | Volatility | $(H_t - L_t) / P_t$ | 0 sessions | Fully available |
| `open_close_return` | Intraday | $P_t / O_t - 1$ | 0 sessions | Fully available |
| `volume_change_1d` | Liquidity | $V_t / V_{t-1} - 1$ | 1 session | Median imputed in training fold |
| `volume_vs_avg20` | Liquidity | $V_t / \text{SMA}_{20}(V) - 1$ | 20 sessions | Median imputed in training fold |
| `trades_vs_avg20` | Liquidity | $N_t / \text{SMA}_{20}(N) - 1$ | 20 sessions | Median imputed in training fold |
| `delivery_pct` | Quality | $\text{Deliverable Qty} / \text{Traded Qty} \times 100$ | 0 sessions | Fully available |
| `rsi_14` | Oscillator | $100 - (100 / (1 + \\text{RS}_{14}))$ | 14 sessions | Median imputed in training fold |
| `day_of_week` | Calendar | Integer 0 (Mon) to 4 (Fri) | 0 sessions | Fully available |
| `month` | Calendar | Integer 1 (Jan) to 12 (Dec) | 0 sessions | Fully available |

*Note: All price ($P, H, L, O$) and volume ($V$) series are corporate-action adjusted backwards to prevent artificial jumps from stock splits, bonus issues, and demergers.*
"""
    with open(FEATURE_DEFINITIONS_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def run_pipeline() -> pd.DataFrame:
    """Executes the complete data preparation and auditing workflow."""
    print("=" * 70)
    print("EXECUTING DATA INGESTION, AUDIT & FEATURE PIPELINE")
    print("=" * 70)

    # 1. Ingestion
    print("\n1. Ingesting equity dataset...")
    df_raw = load_raw_data()
    print(f"   Loaded {len(df_raw):,} rows across {df_raw['Symbol'].nunique()} symbols.")

    # 2. Quality Audit
    print("\n2. Running data quality audit...")
    audit = run_data_quality_audit(df_raw)
    print(f"   Duplicate keys: {audit['duplicate_symbol_date_keys']}")
    print(f"   OHLC violations: {audit['ohlc_inconsistencies']}")
    print(f"   Date range: {audit['date_min']} to {audit['date_max']}")

    # 3. Corporate Action Adjustments
    print("\n3. Applying corporate-action price and volume adjustments...")
    df_adj = apply_corporate_actions(df_raw)
    print(f"   Adjusted {len(CONFIRMED_CORPORATE_ACTIONS)} confirmed corporate actions.")
    print(f"   Register saved to: {CORPORATE_ACTION_REGISTER_PATH}")

    # 4. Feature Engineering
    print("\n4. Engineering leakage-free predictive features...")
    df_feat = compute_features(df_adj)
    print(f"   Engineered {len(FEATURE_COLUMNS)} predictive features.")

    # 5. Target Construction
    print(f"\n5. Constructing {HORIZON_SESSIONS}-session forward target...")
    df_analytical = construct_target(df_feat, horizon=HORIZON_SESSIONS)

    # 6. Target Alignment Audit
    print("\n6. Running target alignment audit...")
    target_audit = audit_target_alignment(df_analytical, horizon=HORIZON_SESSIONS)
    print(f"   Target distribution: Class 1 = {target_audit['class_1_count']} ({target_audit['class_1_ratio']:.1%}), "
          f"Class 0 = {target_audit['class_0_count']} ({1 - target_audit['class_1_ratio']:.1%})")
    print(f"   Ties count: {target_audit['ties_count']}")
    print(f"   Per-symbol alignment verified: {target_audit['target_per_symbol_alignment_verified']}")

    # 7. Documentation & Persistence
    print("\n7. Saving analytical dataset and documentation...")
    save_feature_definitions()
    df_analytical.to_csv(ANALYTICAL_DATASET_PATH, index=False)
    print(f"   Analytical dataset saved to: {ANALYTICAL_DATASET_PATH}")
    print(f"   Feature definitions saved to: {FEATURE_DEFINITIONS_PATH}")
    print("=" * 70)
    print("DATA PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 70)

    return df_analytical


if __name__ == "__main__":
    run_pipeline()
