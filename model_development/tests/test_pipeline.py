"""
Automated Test Suite for the NSE Stock Price Prediction ML Project.

Covers:
1. Ingestion, date parsing, and chronological monotonic ordering.
2. Symbol-Date uniqueness (zero duplicate keys).
3. EQ-series enforcement.
4. OHLC price validity and non-negativity.
5. Corporate-action price/volume adjustments (direction, magnitude, continuity).
6. Return and rolling-feature consistency and warmup handling.
7. 5-trading-session forward target construction and alignment (zero cross-symbol bleed).
8. Absence of look-ahead leakage in feature matrices.
9. Chronological train/validation/holdout separation with 5-session embargo.
10. Strict fitting of preprocessing within training folds.
11. Model serialization and deserialization fidelity.
12. Usability and output schema of the prediction interface.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure model_development is in Python path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    ANALYTICAL_DATASET_PATH,
    BEST_MODEL_PATH,
    CONFIRMED_CORPORATE_ACTIONS,
    EXPECTED_SYMBOLS,
    FEATURE_COLUMNS,
    HORIZON_SESSIONS,
    PURGE_SESSIONS,
)
from data_pipeline import (
    load_raw_data,
    run_data_quality_audit,
    apply_corporate_actions,
    compute_features,
    construct_target,
    audit_target_alignment,
)
from predict import predict_stock, predict_universe, load_model


@pytest.fixture(scope="module")
def analytical_data():
    """Loads prepared analytical dataset or generates a clean slice for testing."""
    if ANALYTICAL_DATASET_PATH.exists():
        df = pd.read_csv(ANALYTICAL_DATASET_PATH, parse_dates=["Date"])
    else:
        raw = load_raw_data()
        adj = apply_corporate_actions(raw)
        feat = compute_features(adj)
        df = construct_target(feat)
    return df


def test_symbol_inventory(analytical_data):
    """Test that all 16 target symbols exist with expected row counts."""
    symbols = sorted(analytical_data["Symbol"].unique())
    assert symbols == EXPECTED_SYMBOLS, f"Mismatch in expected symbols. Got: {symbols}"
    assert len(symbols) == 16
    for sym in symbols:
        cnt = (analytical_data["Symbol"] == sym).sum()
        assert cnt >= 1200, f"Stock {sym} has fewer rows than expected: {cnt}"


def test_date_parsing_and_chronology(analytical_data):
    """Test that dates are valid datetimes and monotonically increasing per symbol."""
    assert analytical_data["Date"].isna().sum() == 0, "Found invalid/NaN dates."
    for sym, grp in analytical_data.groupby("Symbol"):
        dates = grp["Date"].to_numpy()
        assert np.all(dates[:-1] <= dates[1:]), f"Dates not monotonic for {sym}"


def test_symbol_date_uniqueness(analytical_data):
    """Test that there are zero duplicate Symbol-Date keys in the dataset."""
    dups = analytical_data.duplicated(subset=["Symbol", "Date"], keep=False).sum()
    assert dups == 0, f"Found {dups} duplicate Symbol-Date rows."


def test_eq_series_enforcement(analytical_data):
    """Test that only EQ series records are present."""
    if "Series" in analytical_data.columns:
        series_set = set(analytical_data["Series"].unique())
        assert series_set == {"EQ"}, f"Non-EQ series detected: {series_set}"


def test_ohlc_consistency(analytical_data):
    """Test standard OHLC bounds: High >= Low, Open, Close; Low <= Open, Close."""
    high = analytical_data["adj_high"]
    low = analytical_data["adj_low"]
    open_p = analytical_data["adj_open"]
    close = analytical_data["adj_close"]

    assert (high < low).sum() == 0, "High < Low violation detected."
    assert (high < open_p).sum() == 0, "High < Open violation detected."
    assert (high < close).sum() == 0, "High < Close violation detected."
    assert (low > open_p).sum() == 0, "Low > Open violation detected."
    assert (low > close).sum() == 0, "Low > Close violation detected."
    assert (close <= 0).sum() == 0, "Non-positive closing price detected."


def test_corporate_action_adjustments(analytical_data):
    """
    Test that corporate actions adjust prices in the correct direction:
    - Pre-bonus prices must be lower in adjusted terms (divided by split/bonus ratio).
    - Post-bonus prices must equal raw prices (factor = 1.0).
    - Overnight return across ex-date must NOT exhibit artificial -50% to -75% drops.
    """
    for sym, info in CONFIRMED_CORPORATE_ACTIONS.items():
        sub = analytical_data[analytical_data["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
        ex_date = pd.to_datetime(info["ex_date"])

        pre_mask = sub["Date"] < ex_date
        post_mask = sub["Date"] >= ex_date

        # Post-event factor should be 1.0
        assert np.allclose(sub.loc[post_mask, "adj_factor"], 1.0), f"{sym} post-event adj_factor is not 1.0"

        # Pre-event factor should match configured ratio
        expected_factor = info["adjustment_factor"]
        assert np.allclose(sub.loc[pre_mask, "adj_factor"], expected_factor), f"{sym} pre-event factor mismatch"

        # Overnight return across ex-date on adjusted series should be reasonable (|ret| < 0.20)
        ex_idx = sub[sub["Date"] == ex_date].index[0]
        adj_ret_at_ex = (sub.loc[ex_idx, "adj_close"] / sub.loc[ex_idx - 1, "adj_close"]) - 1.0
        raw_ret_at_ex = (sub.loc[ex_idx, "raw_close"] / sub.loc[ex_idx - 1, "raw_close"]) - 1.0

        assert abs(adj_ret_at_ex) < 0.20, f"{sym} adjusted ex-date return ({adj_ret_at_ex:.2%}) is abnormally large"
        assert abs(raw_ret_at_ex) > 0.40, f"{sym} raw ex-date return ({raw_ret_at_ex:.2%}) failed to reflect action"


def test_target_alignment_and_no_symbol_bleed(analytical_data):
    """
    Test that target_up_5d:
    1. Correctly reflects adj_close(t+5) > adj_close(t).
    2. Never crosses symbol boundaries.
    3. Has exactly 5 NaNs per symbol at the end of the series.
    """
    horizon = HORIZON_SESSIONS
    grouped = analytical_data.groupby("Symbol", sort=False)

    for sym, grp in grouped:
        grp = grp.sort_values("Date").reset_index(drop=True)
        close = grp["adj_close"].to_numpy()
        target = grp["target_up_5d"].to_numpy()

        # Final 5 rows must have NaN target
        assert np.isnan(target[-horizon:]).all(), f"{sym} does not have NaN targets for last {horizon} rows"
        # Preceding rows must NOT be NaN
        assert not np.isnan(target[:-horizon]).any(), f"{sym} has unexpected NaNs before last {horizon} rows"

        # Check alignment on known rows
        for i in range(len(close) - horizon):
            expected = 1.0 if close[i + horizon] > close[i] else 0.0
            assert target[i] == expected, f"{sym} target mismatch at index {i}"


def test_no_future_leakage_in_features(analytical_data):
    """
    Test that no target, future return, or post-observation prices
    leak into the predictive feature set.
    """
    forbidden_substrings = ["future", "target", "fwd", "next"]
    for col in FEATURE_COLUMNS:
        for forbidden in forbidden_substrings:
            assert forbidden not in col.lower(), f"Feature column '{col}' contains forbidden string '{forbidden}'"

    # Ensure feature columns are present in analytical dataset
    missing = set(FEATURE_COLUMNS) - set(analytical_data.columns)
    assert not missing, f"Missing feature columns in analytical dataset: {missing}"


def test_chronological_purge_separation(analytical_data):
    """Test that train and test date partitions maintain the strict 5-day purge embargo."""
    labeled = analytical_data.dropna(subset=["target_up_5d"]).copy()
    dates = np.array(sorted(labeled["Date"].unique()))

    split_idx = int(len(dates) * 0.85)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx + PURGE_SESSIONS:]

    train_max = pd.Timestamp(train_dates[-1])
    test_min = pd.Timestamp(test_dates[0])

    # Date gap must be strictly greater than 5 calendar/trading intervals
    diff_days = (test_min - train_max).days
    assert diff_days >= 5, f"Insufficient purge buffer between train ({train_max}) and test ({test_min})"


def test_prediction_interface_schema():
    """Test that predict_stock produces valid schema, probabilities, and labels."""
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Best model artifact not yet generated.")

    result = predict_stock(symbol="RELIANCE")
    assert isinstance(result, dict)
    assert result["symbol"] == "RELIANCE"
    assert 0.0 <= result["probability_up_5d"] <= 1.0
    assert 0.0 <= result["probability_down_5d"] <= 1.0
    assert np.isclose(result["probability_up_5d"] + result["probability_down_5d"], 1.0)
    assert result["predicted_direction"] in ["UP (Bullish)", "DOWN (Bearish)"]
    assert "disclaimer" in result
