"""
Execution of Frozen Research Protocol Experiments
Protocol: "Do Fundamental Financial Indicators Improve ML-Based Prediction of Three-Month NSE Stock Returns?"
Version: 1.0 (24 October 2026)

This script executes:
1. Feature extraction for 3-month return forecasting at monthly frequency (Market-Only).
2. Point-in-time XBRL fundamental indicator extraction and as-of alignment.
3. Frozen model benchmarking on full 16-stock universe (Baselines, Ridge, Random Forest, HistGradientBoosting).
4. Matched primary comparison: Market-Only (A) vs Fundamentals-Only (B) vs Combined (C).
5. Feature-group ablations (Valuation, Profitability, Growth, Leverage, Market).
6. Paired Date-Level Block Bootstrap (L=3 months, B=2,000 replications) for dependence-aware uncertainty.
7. Machine-readable artifact generation (predictions, metrics JSON/CSV, figures, configs, logs).
"""

import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# -----------------------------------------------------------------------------
# Configuration and Directories
# -----------------------------------------------------------------------------
WORKSPACE_ROOT = Path("c:/Users/Adity/OneDrive/Desktop/IML_Research_Project")
EXPERIMENT_DIR = WORKSPACE_ROOT / "protocol_experiments"
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CONFIGS_DIR = OUTPUTS_DIR / "configs"

for d in [PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = EXPERIMENT_DIR / "protocol_execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ProtocolExperiment")

RANDOM_STATE = 42
BOOTSTRAP_REPLICATIONS = 2000
BLOCK_LENGTH_MONTHS = 3  # Matches 3-month overlapping return horizon

# -----------------------------------------------------------------------------
# Step 1: Ingestion and Monthly Market Data Construction
# -----------------------------------------------------------------------------
def load_and_construct_monthly_market_data():
    logger.info("Loading daily analytical dataset...")
    daily_path = WORKSPACE_ROOT / "IML_project" / "model_development" / "outputs" / "datasets" / "analytical_dataset.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing analytical dataset at {daily_path}")
    
    daily_df = pd.read_csv(daily_path, parse_dates=["Date"])
    daily_df = daily_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    
    # Calculate daily returns
    daily_df["ret_daily"] = daily_df.groupby("Symbol")["adj_close"].pct_change()
    
    # Compute rolling lookback features on daily grid
    grouped = daily_df.groupby("Symbol")
    
    daily_df["roll_vol_21"] = grouped["ret_daily"].transform(lambda x: x.rolling(21, min_periods=15).std() * np.sqrt(252))
    daily_df["roll_vol_63"] = grouped["ret_daily"].transform(lambda x: x.rolling(63, min_periods=45).std() * np.sqrt(252))
    daily_df["roll_vol_126"] = grouped["ret_daily"].transform(lambda x: x.rolling(126, min_periods=90).std() * np.sqrt(252))
    
    turnover = daily_df["adj_close"] * daily_df["adj_volume"]
    daily_df["turnover"] = turnover
    daily_df["roll_turnover_21"] = grouped["turnover"].transform(lambda x: x.rolling(21, min_periods=15).mean())
    daily_df["roll_turnover_63"] = grouped["turnover"].transform(lambda x: x.rolling(63, min_periods=45).mean())
    daily_df["vol_volume_21"] = grouped["adj_volume"].transform(lambda x: x.rolling(21, min_periods=15).std() / (x.rolling(21, min_periods=15).mean() + 1e-8))
    daily_df["median_traded_val_63"] = grouped["turnover"].transform(lambda x: x.rolling(63, min_periods=45).median())
    
    if "delivery_pct" in daily_df.columns:
        daily_df["delivery_pct_21"] = grouped["delivery_pct"].transform(lambda x: x.rolling(21, min_periods=15).mean())
    else:
        daily_df["delivery_pct_21"] = 50.0

    # Extract month-end sessions
    daily_df["YearMonth"] = daily_df["Date"].dt.to_period("M")
    monthly_df = daily_df.groupby(["Symbol", "YearMonth"]).last().reset_index()
    monthly_df = monthly_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    
    # Month-lagged returns and momentum
    m_grp = monthly_df.groupby("Symbol")
    monthly_df["ret_1m"] = m_grp["adj_close"].pct_change(1)
    monthly_df["ret_3m"] = m_grp["adj_close"].pct_change(3)
    monthly_df["ret_6m"] = m_grp["adj_close"].pct_change(6)
    monthly_df["ret_12m"] = m_grp["adj_close"].pct_change(12)
    
    # Momentum excluding most recent month:
    # mom_6m_ex1m = P(t-1m) / P(t-6m) - 1
    monthly_df["mom_6m_ex1m"] = m_grp["adj_close"].shift(1) / m_grp["adj_close"].shift(6) - 1.0
    monthly_df["mom_12m_ex1m"] = m_grp["adj_close"].shift(1) / m_grp["adj_close"].shift(12) - 1.0
    
    # Trading activity features
    monthly_df["log_turnover_1m"] = np.log1p(np.maximum(monthly_df["roll_turnover_21"], 0.0))
    monthly_df["turnover_change_1m_3m"] = (monthly_df["roll_turnover_21"] / (monthly_df["roll_turnover_63"] + 1e-8)) - 1.0
    monthly_df["vol_volume_1m"] = monthly_df["vol_volume_21"]
    monthly_df["median_traded_val_3m"] = np.log1p(np.maximum(monthly_df["median_traded_val_63"], 0.0))
    monthly_df["delivery_pct_1m"] = monthly_df["delivery_pct_21"]
    monthly_df["vol_1m"] = monthly_df["roll_vol_21"]
    monthly_df["vol_3m"] = monthly_df["roll_vol_63"]
    monthly_df["vol_6m"] = monthly_df["roll_vol_126"]

    # Target: 3-month forward return (P(t+3m) / P(t) - 1.0)
    monthly_df["fwd_date_3m"] = m_grp["Date"].shift(-3)
    monthly_df["fwd_adj_close_3m"] = m_grp["adj_close"].shift(-3)
    monthly_df["target_return_3m"] = (monthly_df["fwd_adj_close_3m"] / monthly_df["adj_close"]) - 1.0

    logger.info(f"Constructed monthly market panel: {len(monthly_df)} rows across {monthly_df['Symbol'].nunique()} symbols.")
    logger.info(f"Matured target returns available: {monthly_df['target_return_3m'].notna().sum()} rows.")
    return monthly_df

# -----------------------------------------------------------------------------
# Step 2: Ingestion and Point-in-Time Fundamental Feature Extraction
# -----------------------------------------------------------------------------
def load_and_construct_pit_fundamentals():
    logger.info("Loading parsed XBRL fundamental filings...")
    xbrl_path = WORKSPACE_ROOT / "NSE_Fundamentals_Collector" / "NSE_Fundamentals_Collector" / "pilot_collection" / "processed" / "parsed_integrated_fundamentals.csv"
    if not xbrl_path.exists():
        raise FileNotFoundError(f"Missing parsed XBRL data at {xbrl_path}")
    
    df = pd.read_csv(xbrl_path)
    df["broadcast_dt"] = pd.to_datetime(df["broadcast_datetime"], format="mixed", errors="coerce")
    df["broadcast_date"] = df["broadcast_dt"].dt.normalize()
    df = df.dropna(subset=["broadcast_date"]).copy()
    
    df = df.rename(columns={"symbol": "Symbol"})
    # Prefer consolidated filings on duplicate dates
    df["is_cons"] = df["is_consolidated"].fillna(False).astype(bool)
    df = df.sort_values(["Symbol", "broadcast_date", "is_cons"]).drop_duplicates(subset=["Symbol", "broadcast_date"], keep="last").reset_index(drop=True)
    
    # Compute fundamental ratios per filing
    df["net_margin"] = df["q_pat"] / (df["q_revenue"] + 1e-8)
    
    # Growth YoY
    df = df.sort_values(["Symbol", "broadcast_date"]).reset_index(drop=True)
    f_grp = df.groupby("Symbol")
    df["revenue_growth_yoy"] = f_grp["q_revenue"].pct_change(4)
    df["pat_growth_yoy"] = f_grp["q_pat"].pct_change(4)
    
    # Solvency / Leverage
    df["debt_to_assets"] = df["bs_total_liabilities"] / (df["bs_total_assets"] + 1e-8)
    df["debt_to_equity"] = df["bs_total_liabilities"] / (df["bs_equity"] + 1e-8)
    
    # Forward fill balance sheet ratios across quarters where only income statement is reported
    df["debt_to_assets"] = f_grp["debt_to_assets"].ffill()
    df["debt_to_equity"] = f_grp["debt_to_equity"].ffill()
    df["bs_equity_ffill"] = f_grp["bs_equity"].ffill()
    df["bs_total_assets_ffill"] = f_grp["bs_total_assets"].ffill()
    
    # Annualized PAT for ROE / ROA proxies
    df["trailing_annual_pat"] = f_grp["q_pat"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    df["roe_proxy"] = df["trailing_annual_pat"] / (df["bs_equity_ffill"] + 1e-8)
    df["roa_proxy"] = df["trailing_annual_pat"] / (df["bs_total_assets_ffill"] + 1e-8)
    
    fundamental_cols = [
        "Symbol", "broadcast_date", "net_margin", "revenue_growth_yoy", "pat_growth_yoy",
        "debt_to_assets", "debt_to_equity", "roe_proxy", "roa_proxy", "trailing_annual_pat", "bs_equity_ffill"
    ]
    clean_funds = df[fundamental_cols].copy()
    logger.info(f"Cleaned point-in-time fundamentals: {len(clean_funds)} filing snapshots across {clean_funds['Symbol'].unique()}.")
    return clean_funds

# -----------------------------------------------------------------------------
# Step 3: Point-in-Time Merging and Feature Group Definitions
# -----------------------------------------------------------------------------
MARKET_FEATURES = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "mom_6m_ex1m", "mom_12m_ex1m",
    "vol_1m", "vol_3m", "vol_6m",
    "log_turnover_1m", "turnover_change_1m_3m", "vol_volume_1m",
    "median_traded_val_3m", "delivery_pct_1m"
]

FUNDAMENTAL_GROUPS = {
    "valuation": ["pe_proxy", "pb_proxy"],
    "profitability": ["net_margin", "roe_proxy", "roa_proxy"],
    "growth": ["revenue_growth_yoy", "pat_growth_yoy"],
    "leverage": ["debt_to_assets", "debt_to_equity"],
    "size": ["size_log_mcap"]
}

def merge_market_and_fundamentals(monthly_market_df, clean_funds):
    # Perform strict backward merge_asof per symbol to guarantee point-in-time alignment
    symbols = clean_funds["Symbol"].unique()
    merged_frames = []
    
    for sym in symbols:
        sym_m = monthly_market_df[monthly_market_df["Symbol"] == sym].sort_values("Date").copy()
        sym_f = clean_funds[clean_funds["Symbol"] == sym].sort_values("broadcast_date").copy()
        
        merged = pd.merge_asof(sym_m, sym_f, left_on="Date", right_on="broadcast_date", by="Symbol", direction="backward")
        
        # Compute price-dependent fundamental ratios using point-in-time stock price
        # Valuation proxies:
        # Trailing PAT per share proxy:
        merged["pe_proxy"] = (merged["adj_close"] * 1e9) / (merged["trailing_annual_pat"] + 1e-8)
        # Clip extreme outliers or negative PE
        merged["pe_proxy"] = np.where(merged["pe_proxy"] > 0, np.clip(merged["pe_proxy"], 1.0, 150.0), np.nan)
        
        merged["pb_proxy"] = (merged["adj_close"] * 1e9) / (merged["bs_equity_ffill"] + 1e-8)
        merged["pb_proxy"] = np.where(merged["pb_proxy"] > 0, np.clip(merged["pb_proxy"], 0.1, 50.0), np.nan)
        
        merged["size_log_mcap"] = merged["log_turnover_1m"]
        merged_frames.append(merged)
        
    matched_df = pd.concat(merged_frames, ignore_index=True)
    matched_df = matched_df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return matched_df

# -----------------------------------------------------------------------------
# Step 4: Metric Calculation and Evaluation Routines
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_train_mean, dates_arr=None):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    
    # Clearly defined out-of-sample R^2 against training historical-mean forecast:
    # R2_oos = 1 - sum((y - y_hat)^2) / sum((y - y_bar_train)^2)
    denom = np.sum((y_true - y_train_mean) ** 2)
    numer = np.sum((y_true - y_pred) ** 2)
    r2_oos = float(1.0 - (numer / (denom + 1e-12)))
    
    # Directional accuracy
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    
    # Cross-sectional Spearman rank correlation (per date, then averaged)
    spearman_corrs = []
    if dates_arr is not None:
        df_eval = pd.DataFrame({"date": dates_arr, "true": y_true, "pred": y_pred})
        for dt, grp in df_eval.groupby("date"):
            if len(grp) >= 2:
                corr, _ = spearmanr(grp["pred"], grp["true"])
                if not np.isnan(corr):
                    spearman_corrs.append(corr)
        rank_ic = float(np.mean(spearman_corrs)) if spearman_corrs else float(spearmanr(y_pred, y_true)[0])
    else:
        rank_ic = float(spearmanr(y_pred, y_true)[0])
        
    return {
        "mae": mae,
        "rmse": rmse,
        "r2_oos": r2_oos,
        "directional_accuracy": dir_acc,
        "rank_ic": rank_ic
    }

# -----------------------------------------------------------------------------
# Step 5: Paired Date-Level Moving-Block Bootstrap
# -----------------------------------------------------------------------------
def run_paired_block_bootstrap(df_holdout, col_pred_combined, col_pred_baseline, block_length=3, n_reps=2000, seed=42):
    """
    Date-Level Moving-Block Bootstrap.
    Resamples blocks of contiguous monthly trading dates.
    Preserves all stocks on each date together (cross-sectional dependence)
    and preserves the 3-month autocorrelation structure across dates.
    """
    rng = np.random.default_rng(seed)
    unique_dates = np.array(sorted(df_holdout["Date"].unique()))
    n_dates = len(unique_dates)
    
    if n_dates < block_length:
        raise ValueError(f"Holdout dates ({n_dates}) fewer than block length ({block_length})")
        
    n_blocks = int(np.ceil(n_dates / block_length))
    possible_starts = n_dates - block_length + 1
    
    diff_mae_list = []
    pct_imp_list = []
    
    y_true = df_holdout["target_return_3m"].to_numpy()
    p_comb = df_holdout[col_pred_combined].to_numpy()
    p_base = df_holdout[col_pred_baseline].to_numpy()
    dates_all = df_holdout["Date"].to_numpy()
    
    for b in range(n_reps):
        # Sample block starting indices
        start_indices = rng.integers(0, possible_starts, size=n_blocks)
        sampled_dates = []
        for s in start_indices:
            sampled_dates.extend(unique_dates[s : s + block_length])
        sampled_dates = sampled_dates[:n_dates]
        
        # Gather rows corresponding to sampled dates
        idx_sample = []
        for d in sampled_dates:
            idx_sample.extend(np.where(dates_all == d)[0])
        idx_sample = np.array(idx_sample)
        
        sub_true = y_true[idx_sample]
        sub_comb = p_comb[idx_sample]
        sub_base = p_base[idx_sample]
        
        mae_comb = np.mean(np.abs(sub_true - sub_comb))
        mae_base = np.mean(np.abs(sub_true - sub_base))
        
        delta_mae = mae_comb - mae_base
        pct_imp = 100.0 * (mae_base - mae_comb) / (mae_base + 1e-12)
        
        diff_mae_list.append(delta_mae)
        pct_imp_list.append(pct_imp)
        
    diff_mae_arr = np.array(diff_mae_list)
    pct_imp_arr = np.array(pct_imp_list)
    
    # Superiority test: H0: Delta_MAE >= 0 (no improvement / combined is worse or equal)
    # vs H1: Delta_MAE < 0 (combined is superior)
    p_superiority = float(np.mean(diff_mae_arr >= 0.0))
    
    return {
        "point_delta_mae": float(np.mean(np.abs(y_true - p_comb)) - np.mean(np.abs(y_true - p_base))),
        "ci_delta_mae_95": [float(np.percentile(diff_mae_arr, 2.5)), float(np.percentile(diff_mae_arr, 97.5))],
        "point_pct_improvement": float(100.0 * (np.mean(np.abs(y_true - p_base)) - np.mean(np.abs(y_true - p_comb))) / np.mean(np.abs(y_true - p_base))),
        "ci_pct_imp_95": [float(np.percentile(pct_imp_arr, 2.5)), float(np.percentile(pct_imp_arr, 97.5))],
        "p_superiority_h0_delta_ge_zero": p_superiority,
        "diff_distribution": [float(x) for x in diff_mae_list]
    }

# -----------------------------------------------------------------------------
# Master Execution Flow
# -----------------------------------------------------------------------------
def run_all_experiments():
    logger.info("=" * 70)
    logger.info("COMMENCING FROZEN PROTOCOL EXPERIMENTS FOR 3-MONTH RETURNS")
    logger.info("=" * 70)
    
    monthly_market_df = load_and_construct_monthly_market_data()
    clean_funds = load_and_construct_pit_fundamentals()
    matched_df = merge_market_and_fundamentals(monthly_market_df, clean_funds)
    
    # -------------------------------------------------------------------------
    # PART A: Full 16-Stock Universe Benchmark (Market-Only & Baselines)
    # -------------------------------------------------------------------------
    logger.info("\n--- PART A: Full 16-Stock Universe Benchmark (Market-Only) ---")
    univ_labeled = monthly_market_df.dropna(subset=["target_return_3m"]).copy()
    
    # Chronological partition
    dates = sorted(univ_labeled["Date"].unique())
    # Reserve last ~25% dates for holdout, with 3-month purge
    holdout_cutoff_idx = int(len(dates) * 0.70)
    purge_months = 3
    
    train_dates_univ = dates[:holdout_cutoff_idx]
    holdout_dates_univ = dates[holdout_cutoff_idx + purge_months:]
    
    train_univ = univ_labeled[univ_labeled["Date"].isin(train_dates_univ)].copy()
    holdout_univ = univ_labeled[univ_labeled["Date"].isin(holdout_dates_univ)].copy()
    
    y_train_mean_univ = float(train_univ["target_return_3m"].mean())
    y_holdout_univ = holdout_univ["target_return_3m"].to_numpy()
    
    logger.info(f"Universe Train period: {train_dates_univ[0].strftime('%Y-%m')} to {train_dates_univ[-1].strftime('%Y-%m')} ({len(train_univ)} rows)")
    logger.info(f"Universe Purge window: {dates[holdout_cutoff_idx].strftime('%Y-%m')} to {dates[holdout_cutoff_idx+purge_months-1].strftime('%Y-%m')} (3 months)")
    logger.info(f"Universe Holdout period: {holdout_dates_univ[0].strftime('%Y-%m')} to {holdout_dates_univ[-1].strftime('%Y-%m')} ({len(holdout_univ)} rows)")
    logger.info(f"Training Historical Mean Return: {y_train_mean_univ:.4f}")
    
    models_to_test = {
        "Zero_Return_Baseline": None,
        "Historical_Mean_Baseline": None,
        "Ridge_Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE))
        ]),
        "Random_Forest_Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE))
        ]),
        "HistGradientBoosting_Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE))
        ])
    }
    
    univ_results = []
    univ_preds_df = holdout_univ[["Symbol", "Date", "target_return_3m"]].copy()
    
    X_train_u = train_univ[MARKET_FEATURES]
    y_train_u = train_univ["target_return_3m"]
    X_holdout_u = holdout_univ[MARKET_FEATURES]
    
    for name, pipe in models_to_test.items():
        if name == "Zero_Return_Baseline":
            preds = np.zeros(len(holdout_univ))
        elif name == "Historical_Mean_Baseline":
            preds = np.full(len(holdout_univ), y_train_mean_univ)
        else:
            pipe.fit(X_train_u, y_train_u)
            preds = pipe.predict(X_holdout_u)
            
        univ_preds_df[name] = preds
        metrics = compute_metrics(y_holdout_univ, preds, y_train_mean_univ, holdout_univ["Date"].to_numpy())
        metrics["model"] = name
        univ_results.append(metrics)
        logger.info(f"Universe | {name:30s} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | OOS R2: {metrics['r2_oos']:.4f} | Dir Acc: {metrics['directional_accuracy']:.2%} | Rank IC: {metrics['rank_ic']:.4f}")
        
    univ_preds_df.to_csv(PREDICTIONS_DIR / "holdout_predictions_protocol_universe.csv", index=False)
    with open(METRICS_DIR / "protocol_metrics_universe.json", "w") as f:
        json.dump(univ_results, f, indent=2)

    # -------------------------------------------------------------------------
    # PART B: Matched Primary Comparison (Market vs Fundamentals vs Combined)
    # -------------------------------------------------------------------------
    logger.info("\n--- PART B: Matched Primary Comparison (Common Verifiable Sample) ---")
    matched_labeled = matched_df.dropna(subset=["target_return_3m"]).copy()
    
    m_dates = sorted(matched_labeled["Date"].unique())
    m_cutoff_idx = int(len(m_dates) * 0.70)
    
    train_dates_m = m_dates[:m_cutoff_idx]
    holdout_dates_m = m_dates[m_cutoff_idx + purge_months:]
    
    train_m = matched_labeled[matched_labeled["Date"].isin(train_dates_m)].copy()
    holdout_m = matched_labeled[matched_labeled["Date"].isin(holdout_dates_m)].copy()
    
    y_train_mean_m = float(train_m["target_return_3m"].mean())
    y_holdout_m = holdout_m["target_return_3m"].to_numpy()
    
    logger.info(f"Matched Train period: {train_dates_m[0].strftime('%Y-%m')} to {train_dates_m[-1].strftime('%Y-%m')} ({len(train_m)} rows across {train_m['Symbol'].unique()})")
    logger.info(f"Matched Holdout period: {holdout_dates_m[0].strftime('%Y-%m')} to {holdout_dates_m[-1].strftime('%Y-%m')} ({len(holdout_m)} rows across {holdout_m['Symbol'].unique()})")
    
    # Feature sets
    all_fund_features = []
    for grp_cols in FUNDAMENTAL_GROUPS.values():
        all_fund_features.extend(grp_cols)
        
    combined_features = MARKET_FEATURES + all_fund_features
    
    feature_sets = {
        "Market_Only": MARKET_FEATURES,
        "Fundamentals_Only": all_fund_features,
        "Combined": combined_features
    }
    
    matched_eval_records = []
    matched_preds_df = holdout_m[["Symbol", "Date", "target_return_3m"]].copy()
    
    # Evaluate across model families
    model_constructors = {
        "Ridge": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE))]),
        "Random_Forest": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE))]),
        "HistGradientBoosting": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE))]),
    }
    
    # Baselines
    zero_preds = np.zeros(len(holdout_m))
    mean_preds = np.full(len(holdout_m), y_train_mean_m)
    matched_preds_df["Zero_Return_Baseline"] = zero_preds
    matched_preds_df["Historical_Mean_Baseline"] = mean_preds
    
    zero_met = compute_metrics(y_holdout_m, zero_preds, y_train_mean_m, holdout_m["Date"].to_numpy())
    zero_met.update({"feature_set": "Baseline", "model": "Zero_Return_Baseline"})
    matched_eval_records.append(zero_met)
    
    mean_met = compute_metrics(y_holdout_m, mean_preds, y_train_mean_m, holdout_m["Date"].to_numpy())
    mean_met.update({"feature_set": "Baseline", "model": "Historical_Mean_Baseline"})
    matched_eval_records.append(mean_met)
    
    for model_name, constructor in model_constructors.items():
        for fset_name, fcols in feature_sets.items():
            pipe = constructor()
            pipe.fit(train_m[fcols], train_m["target_return_3m"])
            preds = pipe.predict(holdout_m[fcols])
            
            col_key = f"{model_name}_{fset_name}"
            matched_preds_df[col_key] = preds
            
            metrics = compute_metrics(y_holdout_m, preds, y_train_mean_m, holdout_m["Date"].to_numpy())
            metrics.update({"feature_set": fset_name, "model": model_name})
            matched_eval_records.append(metrics)
            logger.info(f"Matched | {model_name:20s} | {fset_name:20s} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | OOS R2: {metrics['r2_oos']:.4f} | Dir Acc: {metrics['directional_accuracy']:.2%} | Rank IC: {metrics['rank_ic']:.4f}")
            
    matched_preds_df.to_csv(PREDICTIONS_DIR / "holdout_predictions_protocol_matched.csv", index=False)
    matched_comp_df = pd.DataFrame(matched_eval_records)
    matched_comp_df.to_csv(METRICS_DIR / "protocol_model_comparison.csv", index=False)
    with open(METRICS_DIR / "protocol_metrics_matched.json", "w") as f:
        json.dump(matched_eval_records, f, indent=2)

    # -------------------------------------------------------------------------
    # PART C: Feature Group Ablation Study
    # -------------------------------------------------------------------------
    logger.info("\n--- PART C: Feature-Group Ablation Study ---")
    ablation_sets = {
        "Combined (All Features)": combined_features,
        "Ablation: No Valuation": [c for c in combined_features if c not in FUNDAMENTAL_GROUPS["valuation"]],
        "Ablation: No Profitability": [c for c in combined_features if c not in FUNDAMENTAL_GROUPS["profitability"]],
        "Ablation: No Growth": [c for c in combined_features if c not in FUNDAMENTAL_GROUPS["growth"]],
        "Ablation: No Leverage": [c for c in combined_features if c not in FUNDAMENTAL_GROUPS["leverage"]],
        "Ablation: Market Only (No Fundamentals)": MARKET_FEATURES,
        "Ablation: Fundamentals Only (No Market)": all_fund_features,
    }
    
    ablation_records = []
    # Test ablations with Ridge and HistGradientBoosting
    for ab_name, ab_cols in ablation_sets.items():
        for m_name in ["Ridge", "HistGradientBoosting"]:
            pipe = model_constructors[m_name]()
            pipe.fit(train_m[ab_cols], train_m["target_return_3m"])
            preds = pipe.predict(holdout_m[ab_cols])
            
            met = compute_metrics(y_holdout_m, preds, y_train_mean_m, holdout_m["Date"].to_numpy())
            met.update({"ablation": ab_name, "model": m_name, "num_features": len(ab_cols)})
            ablation_records.append(met)
            logger.info(f"Ablation | {m_name:20s} | {ab_name:40s} (k={len(ab_cols)}) | MAE: {met['mae']:.4f} | OOS R2: {met['r2_oos']:.4f}")
            
    ablation_df = pd.DataFrame(ablation_records)
    with open(METRICS_DIR / "protocol_ablation_metrics.json", "w") as f:
        json.dump(ablation_records, f, indent=2)

    # -------------------------------------------------------------------------
    # PART D: Paired Date-Level Block Bootstrap Uncertainty
    # -------------------------------------------------------------------------
    logger.info("\n--- PART D: Paired Date-Level Block Bootstrap Uncertainty ---")
    bootstrap_results = {}
    
    for m_name in ["Ridge", "Random_Forest", "HistGradientBoosting"]:
        col_comb = f"{m_name}_Combined"
        col_mkt = f"{m_name}_Market_Only"
        
        boot_out = run_paired_block_bootstrap(
            df_holdout=matched_preds_df,
            col_pred_combined=col_comb,
            col_pred_baseline=col_mkt,
            block_length=BLOCK_LENGTH_MONTHS,
            n_reps=BOOTSTRAP_REPLICATIONS,
            seed=RANDOM_STATE
        )
        
        bootstrap_results[m_name] = {
            "point_delta_mae": boot_out["point_delta_mae"],
            "ci_delta_mae_95": boot_out["ci_delta_mae_95"],
            "point_pct_improvement": boot_out["point_pct_improvement"],
            "ci_pct_imp_95": boot_out["ci_pct_imp_95"],
            "p_superiority_h0_delta_ge_zero": boot_out["p_superiority_h0_delta_ge_zero"]
        }
        
        logger.info(f"Bootstrap | {m_name:20s} | Delta MAE: {boot_out['point_delta_mae']:+.4f} (95% CI: [{boot_out['ci_delta_mae_95'][0]:+.4f}, {boot_out['ci_delta_mae_95'][1]:+.4f}])")
        logger.info(f"Bootstrap | {m_name:20s} | Pct Imp:   {boot_out['point_pct_improvement']:+.2f}% (95% CI: [{boot_out['ci_pct_imp_95'][0]:+.2f}%, {boot_out['ci_pct_imp_95'][1]:+.2f}%]) | p-value (H1: Combined Superior): {boot_out['p_superiority_h0_delta_ge_zero']:.4f}")

    with open(METRICS_DIR / "protocol_bootstrap_uncertainty.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    # -------------------------------------------------------------------------
    # PART E: Diagnostic Visualizations
    # -------------------------------------------------------------------------
    logger.info("\n--- PART E: Generating Diagnostic Figures ---")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Delta MAE Bootstrap Distribution
    plt.figure(figsize=(8, 5))
    for m_name, color in zip(["Ridge", "HistGradientBoosting"], ["#2b8cbe", "#e41a1c"]):
        boot_out = run_paired_block_bootstrap(matched_preds_df, f"{m_name}_Combined", f"{m_name}_Market_Only", block_length=3, n_reps=2000, seed=42)
        sns.kdeplot(boot_out["diff_distribution"], label=f"{m_name} (Median: {np.median(boot_out['diff_distribution']):+.4f})", lw=2, color=color)
    plt.axvline(0.0, color="black", linestyle="--", alpha=0.7, label="Zero Difference (No Improvement)")
    plt.title(r"Paired Block Bootstrap Distribution of $\Delta\mathrm{MAE}$ (Combined minus Market-Only)", fontsize=12, fontweight="bold")
    plt.xlabel(r"$\Delta\mathrm{MAE} = \mathrm{MAE}_{\mathrm{combined}} - \mathrm{MAE}_{\mathrm{market}}$ (Negative favours Combined)")
    plt.ylabel("Bootstrap Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_delta_mae_distribution.png", dpi=300)
    plt.close()
    
    # 2. Feature Group Ablation Bar Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ablation_df, x="mae", y="ablation", hue="model", palette="Set2")
    plt.axvline(mean_met["mae"], color="red", linestyle="--", alpha=0.6, label="Historical Mean Baseline MAE")
    plt.title("Feature-Group Ablation Impact on 3-Month Return Out-of-Sample MAE", fontsize=12, fontweight="bold")
    plt.xlabel("Holdout MAE (Lower is Better)")
    plt.ylabel("Feature Set Configuration")
    plt.xlim(0.05, 0.20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_feature_group_ablation.png", dpi=300)
    plt.close()
    
    # 3. Predicted vs Realized Scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(matched_preds_df["Ridge_Market_Only"], matched_preds_df["target_return_3m"], alpha=0.7, color="#2b8cbe", edgecolors="k", s=60)
    axes[0].plot([-0.3, 0.5], [-0.3, 0.5], "r--", alpha=0.5, label="Perfect Forecast")
    axes[0].set_title("Ridge: Market-Only Features", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted 3-Month Return")
    axes[0].set_ylabel("Actual Realized 3-Month Return")
    axes[0].legend()
    
    axes[1].scatter(matched_preds_df["Ridge_Combined"], matched_preds_df["target_return_3m"], alpha=0.7, color="#31a354", edgecolors="k", s=60)
    axes[1].plot([-0.3, 0.5], [-0.3, 0.5], "r--", alpha=0.5, label="Perfect Forecast")
    axes[1].set_title("Ridge: Combined Features", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted 3-Month Return")
    axes[1].set_ylabel("Actual Realized 3-Month Return")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_predicted_vs_realized.png", dpi=300)
    plt.close()
    
    # 4. Full Universe Model Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    u_df = pd.DataFrame(univ_results)
    sns.barplot(data=u_df, x="model", y="mae", palette="Blues_r")
    plt.title("Full 16-Stock Universe Benchmark: Holdout MAE (3-Month Returns)", fontsize=12, fontweight="bold")
    plt.xlabel("Model Architecture")
    plt.ylabel("Holdout MAE (Decimal Return)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_universe_model_comparison.png", dpi=300)
    plt.close()
    
    logger.info(f"All figures saved to {FIGURES_DIR}")
    
    # Save configuration manifest
    config_manifest = {
        "protocol_name": "Do Fundamental Financial Indicators Improve ML-Based Prediction of Three-Month NSE Stock Returns?",
        "protocol_version": "1.0 (24 October 2026)",
        "prediction_frequency": "Monthly (Last trading session of month)",
        "target_horizon": "3 calendar months (~63 trading sessions)",
        "random_state": RANDOM_STATE,
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_block_length_months": BLOCK_LENGTH_MONTHS,
        "universe_symbols_evaluated": sorted(list(monthly_market_df["Symbol"].unique())),
        "matched_symbols_with_pit_fundamentals": sorted(list(clean_funds["Symbol"].unique())),
        "market_features": MARKET_FEATURES,
        "fundamental_groups": FUNDAMENTAL_GROUPS,
        "purge_window_months": purge_months
    }
    with open(CONFIGS_DIR / "protocol_experiment_config.json", "w") as f:
        json.dump(config_manifest, f, indent=2)
        
    logger.info("=" * 70)
    logger.info("FROZEN PROTOCOL EXPERIMENTS SUCCESSFULLY COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    run_all_experiments()
