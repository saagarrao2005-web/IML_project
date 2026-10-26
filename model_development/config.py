"""
Configuration module for the NSE Stock Price Prediction ML Project.

Centralizes all paths, experiment settings, corporate action records,
feature column definitions, and model hyperparameters to ensure reproducibility
and prevent magic constants across the codebase.
"""

from pathlib import Path

# ==============================================================================
# 1. DIRECTORY PATHS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Input data locations
RAW_DATA_DIR = PROJECT_ROOT / "cleaned_dataset"
if not RAW_DATA_DIR.exists():
    # Case-insensitive fallback for Windows filesystems
    RAW_DATA_DIR = PROJECT_ROOT / "cleaned_Dataset"

EQ_DATA_DIR = BASE_DIR / "outputs" / "equity_series_only"

# Output artifact locations
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

for directory in [DATASETS_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Key analytical file paths
ANALYTICAL_DATASET_PATH = DATASETS_DIR / "analytical_dataset.csv"
CORPORATE_ACTION_REGISTER_PATH = REPORTS_DIR / "corporate_action_register.csv"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_audit_report.json"
TARGET_ALIGNMENT_REPORT_PATH = REPORTS_DIR / "target_alignment_report.json"
FEATURE_DEFINITIONS_PATH = REPORTS_DIR / "feature_definitions.md"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
WALK_FORWARD_RESULTS_PATH = REPORTS_DIR / "walk_forward_results.csv"
WALK_FORWARD_SUMMARY_PATH = REPORTS_DIR / "walk_forward_summary.csv"
PER_STOCK_EVAL_PATH = REPORTS_DIR / "per_stock_evaluation.csv"
HOLDOUT_PREDICTIONS_PATH = PREDICTIONS_DIR / "holdout_predictions.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# ==============================================================================
# 2. EXPERIMENT PARAMETERS
# ==============================================================================

# Target horizon: 5 trading sessions ahead (observation t+5 vs observation t)
HORIZON_SESSIONS = 5

# Purging/embargo buffer to eliminate label leakage from overlapping forward windows
PURGE_SESSIONS = 5

# Deterministic random seed
RANDOM_STATE = 42

# Final untouched holdout fraction (chronological split on distinct trading dates)
HOLDOUT_FRACTION = 0.15

# Walk-forward expanding window folds prior to holdout:
# Each fold specifies (train_end_fraction, val_end_fraction) relative to pre-holdout dates
WALK_FORWARD_SPECS = [
    (0.50, 0.65),
    (0.65, 0.80),
    (0.80, 1.00),
]

# Universe of 16 target NSE equities
EXPECTED_SYMBOLS = sorted([
    "ABB", "ADANIGREEN", "APOLLOHOSP", "HDFCBANK", "INFY", "MARUTI",
    "MOTILALOFS", "POWERINDIA", "RELIANCE", "SBIN", "SIEMENS",
    "SOLARINDS", "TCS", "TITAN", "ULTRACEMCO", "YESBANK"
])

# ==============================================================================
# 3. VERIFIED CORPORATE ACTIONS REGISTER
# ==============================================================================

# Primary-source verified corporate actions that cause mechanical price changes
# Ratio represents the multiplier applied to pre-ex-date historical prices
# so that historical prices are on the same per-share basis as post-action prices.
# Pre-ex-date volumes are multiplied by (1.0 / ratio) to preserve total turnover.
CONFIRMED_CORPORATE_ACTIONS = {
    "MOTILALOFS": {
        "ex_date": "2024-06-10",
        "action_type": "Bonus Issue",
        "ratio_details": "3:1 (3 bonus shares for every 1 existing share; 1 -> 4)",
        "adjustment_factor": 0.25,  # 1 / (1 + 3) = 0.25
        "source": "NSE/BSE Corporate Filings; Trendlyne corporate actions registry",
        "status": "VERIFIED",
        "notes": "Pre-event prices multiplied by 0.25; volumes multiplied by 4.0",
    },
    "RELIANCE": {
        "ex_date": "2024-10-28",
        "action_type": "Bonus Issue",
        "ratio_details": "1:1 (1 bonus share for every 1 existing share; 1 -> 2)",
        "adjustment_factor": 0.50,  # 1 / (1 + 1) = 0.50
        "source": "NSE Circular; RIL Investor Relations Announcement 28-Oct-2024",
        "status": "VERIFIED",
        "notes": "Pre-event prices multiplied by 0.50; volumes multiplied by 2.0",
    },
    "SIEMENS": {
        "ex_date": "2025-04-07",
        "action_type": "Demerger",
        "ratio_details": "Spin-off of Energy Business into Siemens Energy India Limited (1:1 entitlement)",
        # Pre-open price discovery: Siemens Energy ~2478.20, Siemens Limited base price ~2449.95 vs 4928.15 cum-close
        "adjustment_factor": 2449.95 / 4928.15,  # ~0.4971338
        "source": "NSE Demerger Circular; Siemens Limited Notice to Shareholders",
        "status": "VERIFIED",
        "notes": "Demerger of Energy Business. Base discovery price 2449.95 relative to cum-close 4928.15",
    },
    "HDFCBANK": {
        "ex_date": "2025-08-26",
        "action_type": "Bonus Issue",
        "ratio_details": "1:1 (1 bonus share for every 1 existing share; 1 -> 2)",
        "adjustment_factor": 0.50,  # 1 / (1 + 1) = 0.50
        "source": "NSE Circular; HDFC Bank Exchange Disclosures August 2025",
        "status": "VERIFIED",
        "notes": "Pre-event prices multiplied by 0.50; volumes multiplied by 2.0",
    },
}

# Verified market moves that were flagged by audits but are NOT corporate actions
VERIFIED_MARKET_MOVES = [
    {
        "symbol": "ADANIGREEN",
        "dates": ["2023-01-27", "2023-01-30"],
        "reason": "Hindenburg Research report publication selloff; lower circuits hit. Pure market event.",
    },
    {
        "symbol": "ADANIGREEN",
        "dates": ["2023-12-04", "2023-12-05", "2023-12-06"],
        "reason": "Adani group sharp relief rally following Supreme Court hearings and state election results.",
    },
    {
        "symbol": "ADANIGREEN",
        "dates": ["2024-06-04"],
        "reason": "Market-wide crash on Indian General Election (Lok Sabha 2024) vote counting day.",
    },
    {
        "symbol": "ADANIGREEN",
        "dates": ["2024-11-21", "2024-11-29"],
        "reason": "US DOJ indictment announcement selloff followed by clarification rebound.",
    },
    {
        "symbol": "POWERINDIA",
        "dates": ["2025-01-30"],
        "reason": "Q3 FY25 financial results blowout (+498% YoY net profit surge), stock locked in 20% upper circuit.",
    },
    {
        "symbol": "YESBANK",
        "dates": ["2024-02-07"],
        "reason": "Q3 earnings momentum (+349% YoY profit) and market speculation regarding institutional stake sale.",
    },
    {
        "symbol": "ABB, ADANIGREEN, MOTILALOFS, SBIN, SIEMENS",
        "dates": ["2024-06-04"],
        "reason": "Systemic market crash across Indian equities on General Election counting day.",
    },
]

# ==============================================================================
# 4. FEATURE DEFINITIONS
# ==============================================================================

# Predictor features strictly available at close of session t
FEATURE_COLUMNS = [
    # Historical multi-horizon returns
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    # Moving average distance ratios
    "close_vs_sma_5",
    "close_vs_sma_10",
    "close_vs_sma_20",
    "close_vs_sma_50",
    # Volatility indicators
    "volatility_10d",
    "volatility_20d",
    # Intraday structure
    "daily_range_pct",
    "open_close_return",
    # Volume and trade activity
    "volume_change_1d",
    "volume_vs_avg20",
    "trades_vs_avg20",
    "delivery_pct",
    # Relative strength index
    "rsi_14",
    # Calendar features
    "day_of_week",
    "month",
]

# Non-predictor metadata columns
METADATA_COLUMNS = [
    "Symbol",
    "Date",
    "Series",
    "raw_close",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_volume",
    "adj_close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_volume",
    "adj_factor",
    "future_adj_close_5d",
    "future_return_5d",
    "target_up_5d",
]
