"""
Prediction Interface Module for the NSE Stock Price Prediction ML Project.

Provides a clean CLI and programmatic interface for generating directional forecasts
on NSE stocks using the fitted and verified machine learning pipeline.

Important Disclaimers:
- Outputs are statistical model estimates, NOT financial guarantees or investment advice.
- Forecasts predict whether the equity closing price 5 trading observations ahead will be higher
  than the current observation's close.
- Features are strictly computed from completed daily market data available up through the close.
"""

from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
import pandas as pd

from config import (
    BEST_MODEL_PATH,
    ANALYTICAL_DATASET_PATH,
    FEATURE_COLUMNS,
    EXPECTED_SYMBOLS,
    HORIZON_SESSIONS,
)


def load_model(model_path: Path = BEST_MODEL_PATH):
    """Loads the serialized trained pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}.\n"
            "Please run train_evaluate.py or run_project.py first."
        )
    return joblib.load(model_path)


def predict_stock(
    symbol: str,
    date: str = None,
    dataset_path: Path = ANALYTICAL_DATASET_PATH,
    model=None
) -> dict:
    """
    Generates a 5-trading-session directional forecast for a given stock symbol.

    Parameters:
    - symbol: Stock ticker (e.g. 'RELIANCE', 'TCS')
    - date: Optional specific trading date (YYYY-MM-DD). Defaults to the latest available.
    - dataset_path: Path to the prepared analytical dataset.
    - model: Pre-loaded model pipeline object. If None, loaded from disk.

    Returns:
    - Dictionary containing prediction metadata, probabilities, and disclaimers.
    """
    symbol = symbol.strip().upper()
    if symbol not in EXPECTED_SYMBOLS:
        raise ValueError(f"Unknown symbol '{symbol}'. Supported symbols: {EXPECTED_SYMBOLS}")

    if model is None:
        model = load_model()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Analytical dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, parse_dates=["Date"])
    stock_df = df[df["Symbol"] == symbol].sort_values("Date").reset_index(drop=True)

    if stock_df.empty:
        raise ValueError(f"No records found for symbol '{symbol}' in dataset.")

    if date is not None:
        target_date = pd.to_datetime(date).normalize()
        row_match = stock_df[stock_df["Date"] == target_date]
        if row_match.empty:
            available_dates = f"{stock_df['Date'].min().strftime('%Y-%m-%d')} to {stock_df['Date'].max().strftime('%Y-%m-%d')}"
            raise ValueError(f"Date '{date}' not found for {symbol}. Available range: {available_dates}")
        row = row_match.iloc[-1]
    else:
        # Latest available observation
        row = stock_df.iloc[-1]

    # Feature extraction
    feature_values = row[FEATURE_COLUMNS].to_frame().T

    # Generate probability and predicted direction
    if hasattr(model, "predict_proba"):
        prob_up = float(model.predict_proba(feature_values)[0, 1])
    else:
        prob_up = 0.50

    pred_class = int(prob_up >= 0.50)
    direction = "UP (Bullish)" if pred_class == 1 else "DOWN (Bearish)"

    # Confidence interpretation
    margin = abs(prob_up - 0.50)
    if margin < 0.03:
        confidence = "Low / Near-Neutral (Close to 50/50 Chance)"
    elif margin < 0.08:
        confidence = "Moderate Statistical Skew"
    else:
        confidence = "Distinct Directional Skew"

    return {
        "symbol": symbol,
        "as_of_date": row["Date"].strftime("%Y-%m-%d"),
        "current_close_rupees": float(row["raw_close"]),
        "current_adjusted_close": float(row["adj_close"]),
        "prediction_horizon": f"{HORIZON_SESSIONS} trading sessions ahead",
        "predicted_direction": direction,
        "probability_up_5d": round(prob_up, 4),
        "probability_down_5d": round(1.0 - prob_up, 4),
        "confidence_level": confidence,
        "disclaimer": (
            "MODEL ESTIMATE ONLY: This directional forecast is an empirical statistical estimate "
            "derived from technical market indicators. It does NOT predict an exact target price "
            "and does NOT guarantee market performance or profitability."
        )
    }


def predict_universe(date: str = None, dataset_path: Path = ANALYTICAL_DATASET_PATH) -> pd.DataFrame:
    """Generates directional forecasts across all 16 symbols in the universe."""
    model = load_model()
    records = []
    for symbol in EXPECTED_SYMBOLS:
        try:
            res = predict_stock(symbol=symbol, date=date, dataset_path=dataset_path, model=model)
            records.append({
                "Symbol": res["symbol"],
                "As_Of_Date": res["as_of_date"],
                "Current_Close": res["current_close_rupees"],
                "Forecast_Direction": res["predicted_direction"],
                "Probability_Up": res["probability_up_5d"],
                "Confidence": res["confidence_level"],
            })
        except Exception as e:
            records.append({
                "Symbol": symbol,
                "As_Of_Date": date or "LATEST",
                "Current_Close": np.nan,
                "Forecast_Direction": f"UNAVAILABLE: {str(e)}",
                "Probability_Up": np.nan,
                "Confidence": "N/A",
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="NSE Stock Directional Prediction Interface")
    parser.add_argument("--symbol", type=str, default=None, help="Stock ticker (e.g. RELIANCE, INFY)")
    parser.add_argument("--date", type=str, default=None, help="Prediction date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Generate predictions for all 16 stocks")
    args = parser.parse_args()

    print("=" * 65)
    print("NSE 5-DAY STOCK DIRECTIONAL FORECAST INTERFACE")
    print("=" * 65)

    if args.all or args.symbol is None:
        print("\nGenerating forecasts for all 16 universe equities...")
        forecasts = predict_universe(date=args.date)
        print(forecasts.to_string(index=False))
    else:
        res = predict_stock(symbol=args.symbol, date=args.date)
        print(f"\nSymbol:               {res['symbol']}")
        print(f"As of Market Close:   {res['as_of_date']}")
        print(f"Current Close:        INR {res['current_close_rupees']:,.2f}")
        print(f"Prediction Horizon:   {res['prediction_horizon']}")
        print(f"Forecast Direction:   {res['predicted_direction']}")
        print(f"Probability Up (5d):  {res['probability_up_5d']:.1%}")
        print(f"Probability Down (5d):{res['probability_down_5d']:.1%}")
        print(f"Confidence Level:     {res['confidence_level']}")
        print(f"\n{res['disclaimer']}")

    print("=" * 65)


if __name__ == "__main__":
    main()
