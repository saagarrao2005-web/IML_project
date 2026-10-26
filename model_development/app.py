"""
Streamlit Web Dashboard for NSE Stock Price Prediction ML Project.

B.Tech Introduction to Machine Learning (IML) Project
Focus: Directional Equity Forecasting on 16 NSE Stocks using Machine Learning.

This application provides:
1. Market Dashboard: Universe forecasts across all 16 NSE equities.
2. Stock Prediction: Directional forecasts, historical price charts, and recent sessions.
3. Model Performance: Holdout metrics, walk-forward validation, and diagnostic figures.
4. About Project: Academic documentation, feature definitions, and disclaimers.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure model_development is in Python path for robust imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import (
    ANALYTICAL_DATASET_PATH,
    BEST_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    WALK_FORWARD_SUMMARY_PATH,
    PER_STOCK_EVAL_PATH,
    FIGURES_DIR,
    EXPECTED_SYMBOLS,
    HORIZON_SESSIONS,
    CONFIRMED_CORPORATE_ACTIONS,
    FEATURE_COLUMNS,
)
from predict import load_model, predict_stock, predict_universe

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="NSE Stock Price Prediction | B.Tech IML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ── Global resets ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Page header ── */
    .page-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(99,102,241,0.15);
    }
    .page-header h1 {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .page-header p {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e1f36 0%, #181926 100%);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .metric-card-title {
        font-size: 0.72rem;
        color: #818cf8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.2;
    }
    .metric-card-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Accent variants */
    .metric-card.accent-green {
        border-color: rgba(34,197,94,0.25);
    }
    .metric-card.accent-green .metric-card-title {
        color: #4ade80;
    }
    .metric-card.accent-red {
        border-color: rgba(239,68,68,0.25);
    }
    .metric-card.accent-red .metric-card-title {
        color: #f87171;
    }
    .metric-card.accent-amber {
        border-color: rgba(245,158,11,0.25);
    }
    .metric-card.accent-amber .metric-card-title {
        color: #fbbf24;
    }

    /* ── Directional badges ── */
    .badge-up {
        display: inline-block;
        background-color: rgba(34,197,94,0.15);
        color: #4ade80;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 5px 12px;
        border-radius: 6px;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .badge-down {
        display: inline-block;
        background-color: rgba(239,68,68,0.15);
        color: #f87171;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 5px 12px;
        border-radius: 6px;
        border: 1px solid rgba(239,68,68,0.3);
    }

    /* ── Info / Warning callouts ── */
    .callout-info {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(99,102,241,0.03) 100%);
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 18px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.65;
    }
    .callout-info strong {
        color: #e2e8f0;
    }
    .callout-warning {
        background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.03) 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 18px 0;
        font-size: 0.86rem;
        color: #cbd5e1;
        line-height: 1.65;
    }
    .callout-warning strong {
        color: #fbbf24;
    }

    /* ── Probability bar ── */
    .prob-track {
        display: flex;
        height: 30px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(99,102,241,0.2);
        margin: 10px 0;
    }
    .prob-down {
        background: linear-gradient(90deg, #991b1b, #dc2626);
        color: #fecaca;
        font-weight: 700;
        font-size: 0.78rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .prob-up {
        background: linear-gradient(90deg, #16a34a, #22c55e);
        color: #dcfce7;
        font-weight: 700;
        font-size: 0.78rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* ── Forecast distribution strip ── */
    .forecast-strip {
        background: linear-gradient(135deg, #1e1f36 0%, #181926 100%);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }
    .forecast-strip .label {
        font-size: 0.83rem;
        font-weight: 600;
        color: #94a3b8;
    }
    .forecast-strip .bull {
        font-size: 0.92rem;
        font-weight: 700;
        color: #4ade80;
        margin-left: 10px;
    }
    .forecast-strip .bear {
        font-size: 0.92rem;
        font-weight: 700;
        color: #f87171;
    }
    .forecast-strip .divider {
        color: rgba(99,102,241,0.3);
        margin: 0 8px;
    }
    .forecast-strip .meta {
        font-size: 0.82rem;
        color: #64748b;
    }
    .forecast-strip .meta strong {
        color: #a5b4fc;
    }

    /* ── Section headers ── */
    .section-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: #6366f1;
        margin-bottom: 10px;
        margin-top: 28px;
    }

    /* ── Sidebar tweaks ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12131f 0%, #0d0e18 100%);
    }
    .sidebar-brand {
        padding: 8px 0 14px 0;
        border-bottom: 1px solid rgba(99,102,241,0.15);
        margin-bottom: 16px;
    }
    .sidebar-brand .tag {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: #818cf8;
        font-weight: 700;
    }
    .sidebar-brand .title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f1f5f9;
        margin-top: 3px;
        letter-spacing: -0.01em;
    }
    .sidebar-brand .sub {
        font-size: 0.76rem;
        color: #64748b;
    }
    .sidebar-summary {
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.65;
        padding: 4px 0;
    }
    .sidebar-summary strong {
        color: #c7d2fe;
    }
    .sidebar-disclaimer {
        font-size: 0.68rem;
        color: #475569;
        line-height: 1.45;
        font-style: italic;
    }

    /* ── Dataframe styling ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Tab bar tweaks ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & ARTIFACT LOADERS
# ==============================================================================

@st.cache_resource(show_spinner="Loading trained ML pipeline...")
def get_model():
    """Loads and caches the trained pipeline from disk."""
    return load_model(BEST_MODEL_PATH)


@st.cache_data(show_spinner="Loading analytical dataset...")
def get_dataset() -> pd.DataFrame:
    """Loads and caches the analytical dataset."""
    if not ANALYTICAL_DATASET_PATH.exists():
        st.error(f"Dataset not found at {ANALYTICAL_DATASET_PATH}. Please run the pipeline first.")
        return pd.DataFrame()
    df = pd.read_csv(ANALYTICAL_DATASET_PATH, parse_dates=["Date"])
    return df


@st.cache_data(show_spinner="Generating directional forecasts...")
def get_universe_predictions() -> pd.DataFrame:
    """Computes directional forecasts across all 16 universe equities."""
    return predict_universe(dataset_path=ANALYTICAL_DATASET_PATH)


@st.cache_data
def get_evaluation_reports():
    """Loads model evaluation tables from outputs/reports."""
    comp_df = pd.DataFrame()
    wf_df = pd.DataFrame()
    stock_df = pd.DataFrame()

    if MODEL_COMPARISON_PATH.exists():
        comp_df = pd.read_csv(MODEL_COMPARISON_PATH)
    if WALK_FORWARD_SUMMARY_PATH.exists():
        wf_df = pd.read_csv(WALK_FORWARD_SUMMARY_PATH, header=[0, 1], index_col=0)
    if PER_STOCK_EVAL_PATH.exists():
        stock_df = pd.read_csv(PER_STOCK_EVAL_PATH)

    return comp_df, wf_df, stock_df


# ==============================================================================
# HELPER: Render a page header consistently
# ==============================================================================

def _page_header(title: str, subtitle: str, emoji: str = ""):
    prefix = f"{emoji} " if emoji else ""
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{prefix}{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(title, value, subtitle="", accent=""):
    cls = f"metric-card accent-{accent}" if accent else "metric-card"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="metric-card-title">{title}</div>
            <div class="metric-card-value">{value}</div>
            <div class="metric-card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="tag">B.Tech IML Project</div>
            <div class="title">📈 NSE Stock Prediction</div>
            <div class="sub">Directional ML Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠  Market Dashboard",
            "🔍  Stock Prediction",
            "📊  Model Performance",
            "📖  About Project",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-summary">
            <strong>Universe:</strong> {len(EXPECTED_SYMBOLS)} NSE Equities<br>
            <strong>Horizon:</strong> {HORIZON_SESSIONS} Trading Sessions<br>
            <strong>Data:</strong> Oct 2021 – Oct 2026<br>
            <strong>Model:</strong> Logistic Regression (L2)<br>
            <strong>Validation:</strong> Walk-Forward (3 folds)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="sidebar-disclaimer">
            Academic coursework demonstration. Outputs are
            statistical model estimates, not financial or
            investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return page


# ==============================================================================
# PAGE 1: MARKET DASHBOARD
# ==============================================================================

def render_dashboard():
    _page_header(
        "Market Dashboard",
        "Statistical 5-session directional forecasts across 16 NSE equities",
        "🏠",
    )

    preds_df = get_universe_predictions()
    dataset = get_dataset()

    # Compute data-driven summary values
    latest_date_str = "N/A"
    total_records = 0
    if not dataset.empty:
        latest_date_str = dataset["Date"].max().strftime("%d %b %Y")
        total_records = len(dataset)

    # ── KPI cards ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _metric_card("Equities Monitored", len(EXPECTED_SYMBOLS), "NSE Universe Sample")
    with col2:
        _metric_card("Forecast Horizon", f"{HORIZON_SESSIONS} Sessions", f"Direction at t+{HORIZON_SESSIONS} vs t")
    with col3:
        _metric_card("Deployed Model", "Logistic Reg.", "L2-Regularised Baseline")
    with col4:
        _metric_card("Latest Data", latest_date_str, f"{total_records:,} total observations")

    # ── Direction distribution strip ──
    if not preds_df.empty:
        up_count = int(preds_df["Forecast_Direction"].str.contains("UP", na=False).sum())
        down_count = int(preds_df["Forecast_Direction"].str.contains("DOWN", na=False).sum())
        total_preds = len(preds_df)
        avg_up_prob = preds_df["Probability_Up"].mean()
        prob_min = preds_df["Probability_Up"].min()
        prob_max = preds_df["Probability_Up"].max()

        st.markdown(
            f"""
            <div class="forecast-strip">
                <div>
                    <span class="label">Forecast Distribution:</span>
                    <span class="bull">▲ {up_count} Bullish ({up_count/total_preds*100:.0f}%)</span>
                    <span class="divider">│</span>
                    <span class="bear">▼ {down_count} Bearish ({down_count/total_preds*100:.0f}%)</span>
                </div>
                <div class="meta">
                    Range: <strong>{prob_min:.1%}</strong> – <strong>{prob_max:.1%}</strong>
                    &nbsp;·&nbsp; Mean: <strong>{avg_up_prob:.1%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Filters ──
    st.markdown('<div class="section-label">🔎 Filter Predictions</div>', unsafe_allow_html=True)

    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])

    with filter_col1:
        direction_filter = st.selectbox(
            "Direction",
            ["All Directions", "▲ UP (Bullish)", "▼ DOWN (Bearish)"],
            key="dash_dir_filter",
        )
    with filter_col2:
        confidence_filter = st.selectbox(
            "Confidence",
            ["All Confidence Levels", "Low / Near-Neutral", "Moderate Statistical Skew", "Distinct Directional Skew"],
            key="dash_conf_filter",
        )
    with filter_col3:
        search_query = st.text_input(
            "Search Ticker",
            placeholder="e.g. RELIANCE, INFY …",
            key="dash_search",
        ).strip().upper()

    # Apply filters
    display_df = preds_df.copy()
    if direction_filter != "All Directions":
        key = "UP" if "UP" in direction_filter else "DOWN"
        display_df = display_df[display_df["Forecast_Direction"].str.contains(key, na=False)]

    if confidence_filter != "All Confidence Levels":
        display_df = display_df[
            display_df["Confidence"].str.contains(confidence_filter, regex=False, na=False)
        ]

    if search_query:
        display_df = display_df[
            display_df["Symbol"].str.contains(search_query, regex=False, na=False)
        ]

    # ── Prediction Table ──
    st.markdown('<div class="section-label">📋 Prediction Table</div>', unsafe_allow_html=True)

    if display_df.empty:
        st.info("No stocks match the selected filters. Try adjusting your criteria.")
    else:
        table_records = []
        for _, row in display_df.iterrows():
            is_up = "UP" in str(row["Forecast_Direction"])
            prob_up_val = row["Probability_Up"]
            prob_down_val = 1.0 - prob_up_val if pd.notnull(prob_up_val) else np.nan

            table_records.append({
                "Symbol": row["Symbol"],
                "Latest Close": f"₹{row['Current_Close']:,.2f}" if pd.notnull(row["Current_Close"]) else "N/A",
                "Forecast": "▲ UP" if is_up else "▼ DOWN",
                "P(Up)": f"{prob_up_val:.1%}" if pd.notnull(prob_up_val) else "N/A",
                "P(Down)": f"{prob_down_val:.1%}" if pd.notnull(prob_down_val) else "N/A",
                "Confidence": row["Confidence"],
                "As-of Date": row["As_Of_Date"],
            })

        st.dataframe(
            pd.DataFrame(table_records),
            use_container_width=True,
            hide_index=True,
            height=min(600, 38 * (len(table_records) + 1)),
        )

    # CSV Export
    col_down, _ = st.columns([1, 3])
    with col_down:
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Export Predictions (CSV)",
            data=csv_data,
            file_name=f"nse_predictions_{latest_date_str}.csv",
            mime="text/csv",
        )

    # ── Summary callout ──
    if not preds_df.empty:
        top_stock = preds_df.loc[preds_df["Probability_Up"].idxmax()]
        bot_stock = preds_df.loc[preds_df["Probability_Up"].idxmin()]
        near_neutral = int((preds_df["Confidence"].str.contains("Low", na=False)).sum())

        st.markdown(
            f"""
            <div class="callout-info">
                <strong>Summary (as of {latest_date_str}):</strong><br>
                Directional probability estimates range from <strong>{prob_min:.1%}</strong>
                to <strong>{prob_max:.1%}</strong>, avoiding overconfident extremes.<br>
                Highest up-probability: <strong>{top_stock['Symbol']}
                ({top_stock['Probability_Up']:.1%})</strong>.
                Lowest: <strong>{bot_stock['Symbol']}
                ({bot_stock['Probability_Up']:.1%})</strong>.<br>
                {near_neutral} of {total_preds} equities are classified as
                <em>Low / Near-Neutral</em>, reflecting the noise-dominated nature
                of short-term daily returns.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: INDIVIDUAL STOCK PREDICTION
# ==============================================================================

def render_stock_prediction():
    _page_header(
        "Stock Prediction",
        "Individual 5-session directional forecast, price history, and technical indicators",
        "🔍",
    )

    # Stock selector — default to RELIANCE
    default_idx = EXPECTED_SYMBOLS.index("RELIANCE") if "RELIANCE" in EXPECTED_SYMBOLS else 0

    sel_col, _ = st.columns([2, 4])
    with sel_col:
        selected_symbol = st.selectbox(
            "Select NSE Equity",
            EXPECTED_SYMBOLS,
            index=default_idx,
            key="stock_selector",
        )

    dataset = get_dataset()
    if dataset.empty:
        st.error("Dataset not loaded. Cannot display stock details.")
        return

    stock_df = dataset[dataset["Symbol"] == selected_symbol].sort_values("Date").reset_index(drop=True)
    if stock_df.empty:
        st.error(f"No records found for {selected_symbol}.")
        return

    # Generate forecast
    model = get_model()
    try:
        pred_result = predict_stock(selected_symbol, dataset_path=ANALYTICAL_DATASET_PATH, model=model)
    except Exception as e:
        st.error(f"Prediction failed for {selected_symbol}: {e}")
        return

    # ── Forecast KPI row ──
    is_up = "UP" in pred_result["predicted_direction"]
    badge_html = (
        f'<span class="badge-up">▲ {pred_result["predicted_direction"]}</span>'
        if is_up
        else f'<span class="badge-down">▼ {pred_result["predicted_direction"]}</span>'
    )

    st.markdown('<div class="section-label">⚡ Forecast Result</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _metric_card(
            "Latest Available Close",
            f"₹{pred_result['current_close_rupees']:,.2f}",
            f"As of {pred_result['as_of_date']}",
        )

    with col2:
        accent = "green" if is_up else "red"
        st.markdown(
            f"""
            <div class="metric-card accent-{accent}">
                <div class="metric-card-title">Directional Forecast</div>
                <div style="margin-top: 8px; margin-bottom: 4px;">{badge_html}</div>
                <div class="metric-card-subtitle">Target: {HORIZON_SESSIONS} sessions (T+{HORIZON_SESSIONS})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        prob_color = "green" if pred_result["probability_up_5d"] >= 0.5 else "red"
        _metric_card(
            f"Probability Up ({HORIZON_SESSIONS}d)",
            f"{pred_result['probability_up_5d']:.1%}",
            f"Down: {pred_result['probability_down_5d']:.1%}",
            accent=prob_color,
        )

    with col4:
        confidence_short = pred_result["confidence_level"].split("(")[0].strip()
        margin_pct = abs(pred_result["probability_up_5d"] - 0.50)
        _metric_card(
            "Confidence",
            confidence_short,
            f"Margin over 50%: {margin_pct:.1%}",
            accent="amber",
        )

    # ── Probability bar ──
    prob_up = pred_result["probability_up_5d"]
    prob_down = pred_result["probability_down_5d"]
    st.markdown(
        f"""
        <div style="margin: 8px 0 24px 0;">
            <div style="display: flex; justify-content: space-between;
                        font-size: 0.80rem; font-weight: 600; margin-bottom: 5px;">
                <span style="color: #f87171;">▼ DOWN: {prob_down:.1%}</span>
                <span style="color: #64748b;">50 / 50</span>
                <span style="color: #4ade80;">▲ UP: {prob_up:.1%}</span>
            </div>
            <div class="prob-track">
                <div class="prob-down" style="width: {prob_down*100:.1f}%;">
                    {prob_down:.1%}
                </div>
                <div class="prob-up" style="width: {prob_up*100:.1f}%;">
                    {prob_up:.1%}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence interpretation
    st.markdown(
        f"""
        <div class="callout-info">
            <strong>Confidence Interpretation:</strong>
            Probabilities near 50% indicate the model sees no strong directional signal.
            A probability above 53% is labelled <em>Moderate Statistical Skew</em>;
            above 58% is <em>Distinct Directional Skew</em>. This stock's up-probability
            of <strong>{prob_up:.1%}</strong> (margin: {margin_pct:.1%}) is classified as
            <strong>{confidence_short}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Historical price chart ──
    st.markdown(
        f'<div class="section-label">📈 {selected_symbol} — Price History</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2, chart_col3 = st.columns([2, 2, 2])
    with chart_col1:
        lookback = st.selectbox(
            "Lookback Period",
            ["6 Months", "3 Months", "1 Year", "Full History"],
            index=0,
            key="lookback_sel",
        )
    with chart_col2:
        price_col_options = ["Adjusted Close"]
        if "raw_close" in stock_df.columns:
            price_col_options.append("Raw Close")
        series_type = st.radio(
            "Price Series",
            price_col_options,
            horizontal=True,
            key="price_series_sel",
        )
    with chart_col3:
        show_sma = st.checkbox("Overlay 20-Day SMA", value=True, key="sma_check")

    # Filter by lookback
    n_days_map = {
        "3 Months": 63,
        "6 Months": 126,
        "1 Year": 252,
        "Full History": len(stock_df),
    }
    n_days = n_days_map[lookback]
    sub_df = stock_df.tail(n_days).copy()

    target_price_col = "adj_close" if "Adjusted" in series_type else "raw_close"
    if target_price_col not in sub_df.columns:
        target_price_col = "adj_close"

    plot_df = sub_df[["Date", target_price_col]].copy()
    plot_df = plot_df.rename(columns={target_price_col: f"{selected_symbol} Close"})
    plot_df.set_index("Date", inplace=True)

    if show_sma:
        sma_series = sub_df[target_price_col].rolling(20).mean()
        plot_df["20-Day SMA"] = sma_series.values

    st.line_chart(plot_df, use_container_width=True)

    # ── Technical indicators ──
    latest_row = stock_df.iloc[-1]
    st.markdown('<div class="section-label">📐 Latest Session Indicators</div>', unsafe_allow_html=True)

    indicator_available = True
    for col_name in ["rsi_14", "volatility_20d", "return_5d", "volume_vs_avg20"]:
        if col_name not in latest_row.index or pd.isna(latest_row[col_name]):
            indicator_available = False
            break

    if indicator_available:
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            rsi_val = latest_row["rsi_14"]
            if rsi_val > 70:
                rsi_status = "Overbought (>70)"
            elif rsi_val < 30:
                rsi_status = "Oversold (<30)"
            else:
                rsi_status = "Neutral"
            _metric_card("RSI (14)", f"{rsi_val:.1f}", rsi_status)

        with t_col2:
            vol_val = latest_row["volatility_20d"] * 100
            _metric_card("20-Day Volatility", f"{vol_val:.2f}%", "Daily Return Std Dev")

        with t_col3:
            ret5_val = latest_row["return_5d"] * 100
            accent = "green" if ret5_val >= 0 else "red"
            _metric_card("5-Day Return", f"{ret5_val:+.2f}%", "Trailing (Historical)", accent=accent)

        with t_col4:
            vol_ratio = latest_row["volume_vs_avg20"]
            _metric_card(
                "Volume vs 20D Avg",
                f"{vol_ratio:.2f}x",
                f"{'Above' if vol_ratio >= 1.0 else 'Below'} Average",
            )

        st.caption(
            "Technical indicators are descriptive statistics of recent price action. "
            "They do not by themselves demonstrate predictive skill."
        )
    else:
        st.info("Some technical indicator data is unavailable for this stock's latest session.")

    # ── Recent trading sessions table ──
    st.markdown(
        f'<div class="section-label">🗓 Recent 10 Sessions — {selected_symbol}</div>',
        unsafe_allow_html=True,
    )
    recent_10 = stock_df.tail(10).sort_values("Date", ascending=False).copy()

    recent_table = []
    for _, r in recent_10.iterrows():
        ret1 = r["return_1d"] * 100 if pd.notnull(r.get("return_1d")) else 0.0
        ret5 = r["return_5d"] * 100 if pd.notnull(r.get("return_5d")) else 0.0
        row_data = {
            "Date": r["Date"].strftime("%Y-%m-%d"),
            "Close": f"₹{r['raw_close']:,.2f}" if pd.notnull(r.get("raw_close")) else "N/A",
            "Adj Close": f"₹{r['adj_close']:,.2f}" if pd.notnull(r.get("adj_close")) else "N/A",
        }
        if "raw_open" in r.index and pd.notnull(r["raw_open"]):
            row_data["Open"] = f"₹{r['raw_open']:,.2f}"
        if "raw_high" in r.index and pd.notnull(r["raw_high"]):
            row_data["High"] = f"₹{r['raw_high']:,.2f}"
        if "raw_low" in r.index and pd.notnull(r["raw_low"]):
            row_data["Low"] = f"₹{r['raw_low']:,.2f}"
        if "raw_volume" in r.index and pd.notnull(r["raw_volume"]):
            row_data["Volume"] = f"{int(r['raw_volume']):,}"

        row_data["1D Ret"] = f"{ret1:+.2f}%"
        row_data["5D Ret"] = f"{ret5:+.2f}%"

        if "rsi_14" in r.index and pd.notnull(r["rsi_14"]):
            row_data["RSI"] = f"{r['rsi_14']:.1f}"

        recent_table.append(row_data)

    st.dataframe(pd.DataFrame(recent_table), use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown(
        """
        <div class="callout-warning">
            <strong>Disclaimer:</strong> This directional forecast is a statistical model estimate
            derived from point-in-time technical features. It does <strong>not</strong> constitute
            investment advice, trading signals, or price targets. Financial markets are subject to
            risks that statistical models cannot foresee.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 3: MODEL PERFORMANCE
# ==============================================================================

def render_model_performance():
    _page_header(
        "Model Performance",
        "Out-of-sample holdout evaluation, walk-forward validation, and diagnostic figures",
        "📊",
    )

    comp_df, wf_df, stock_df = get_evaluation_reports()

    # ── Top summary cards ──
    col1, col2, col3, col4 = st.columns(4)

    if not comp_df.empty:
        non_dummy = comp_df[comp_df["model"] != "Dummy_Baseline"]
        best_holdout = non_dummy.loc[non_dummy["roc_auc"].idxmax()]
        deployed = comp_df[comp_df["model"] == "Logistic_Regression"]
        deployed_auc = deployed.iloc[0]["roc_auc"] if not deployed.empty else "N/A"
        holdout_n = int(best_holdout["holdout_rows"])
    else:
        best_holdout = None
        deployed_auc = "N/A"
        holdout_n = "N/A"

    with col1:
        if best_holdout is not None:
            _metric_card(
                "Best Holdout ROC-AUC",
                f"{best_holdout['roc_auc']:.4f}",
                best_holdout['model'].replace('_', ' '),
                accent="green",
            )
        else:
            st.info("No holdout evaluation data available.")

    with col2:
        _metric_card("Deployed Pipeline AUC", f"{deployed_auc:.4f}", "Logistic Regression")

    with col3:
        _metric_card("Holdout Set Size", f"{holdout_n:,}", "Chronological test split")

    with col4:
        st.markdown(
            """
            <div class="metric-card accent-amber">
                <div class="metric-card-title">Deployed vs Best Holdout</div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 8px; line-height: 1.55;">
                    Logistic Regression was selected for best
                    walk-forward CV performance &amp;
                    interpretability, not solely holdout AUC.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Key finding callout
    st.markdown(
        """
        <div class="callout-info">
            <strong>Key Empirical Finding:</strong><br>
            Consistent with the <strong>Weak-Form Efficient Market Hypothesis</strong>,
            daily technical indicators alone achieve holdout ROC-AUC values near 0.50
            (chance level). No candidate model demonstrated statistically significant
            predictive skill over random guessing in chronological walk-forward validation.<br><br>
            In academic ML, a scientifically defensible null result under leak-free
            chronological validation is more credible than an overfitted high-accuracy artifact.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Table 1: Holdout comparison ──
    st.markdown('<div class="section-label">1 · Holdout Model Comparison</div>', unsafe_allow_html=True)
    if not comp_df.empty:
        formatted_comp = []
        for _, r in comp_df.iterrows():
            formatted_comp.append({
                "Model": r["model"].replace("_", " "),
                "ROC-AUC": f"{r['roc_auc']:.4f}",
                "Bal. Accuracy": f"{r['balanced_accuracy']:.2%}",
                "Accuracy": f"{r['accuracy']:.2%}",
                "Precision": f"{r['precision']:.2%}",
                "Recall": f"{r['recall']:.2%}",
                "F1": f"{r['f1']:.4f}",
                "PR-AUC": f"{r['pr_auc']:.4f}",
                "Brier": f"{r['brier_score']:.4f}",
            })
        st.dataframe(pd.DataFrame(formatted_comp), use_container_width=True, hide_index=True)
    else:
        st.warning("Holdout evaluation file not found. Run `python run_project.py` to generate.")

    with st.expander("ℹ️  What do these metrics mean?"):
        st.markdown(
            """
            - **ROC-AUC:** Measures ranking ability. 0.50 = random guessing; 1.0 = perfect.
            - **Balanced Accuracy:** Average recall across UP and DOWN classes.
            - **Brier Score:** Mean squared probability error. 0.25 = uninformative (50/50).
            - **PR-AUC:** Precision-Recall AUC, important when class balance matters.
            """
        )

    # ── Table 2: Walk-Forward ──
    st.markdown('<div class="section-label">2 · Walk-Forward Cross-Validation (3 Expanding Folds)</div>', unsafe_allow_html=True)
    if not wf_df.empty:
        formatted_wf = []
        for model_name, row in wf_df.iterrows():
            try:
                formatted_wf.append({
                    "Model": model_name.replace("_", " "),
                    "ROC-AUC": f"{row[('roc_auc', 'mean')]:.4f} ± {row[('roc_auc', 'std')]:.4f}",
                    "Bal. Accuracy": f"{row[('balanced_accuracy', 'mean')]:.4f} ± {row[('balanced_accuracy', 'std')]:.4f}",
                    "Accuracy": f"{row[('accuracy', 'mean')]:.4f} ± {row[('accuracy', 'std')]:.4f}",
                    "PR-AUC": f"{row[('pr_auc', 'mean')]:.4f} ± {row[('pr_auc', 'std')]:.4f}",
                    "Brier": f"{row[('brier_score', 'mean')]:.4f} ± {row[('brier_score', 'std')]:.4f}",
                })
            except (KeyError, TypeError):
                continue
        if formatted_wf:
            st.dataframe(pd.DataFrame(formatted_wf), use_container_width=True, hide_index=True)
    else:
        st.warning("Walk-forward results not found.")

    # ── Table 3: Per-Stock ──
    st.markdown('<div class="section-label">3 · Per-Stock Holdout Performance</div>', unsafe_allow_html=True)
    if not stock_df.empty:
        formatted_stocks = []
        for _, r in stock_df.iterrows():
            auc_str = f"{r['roc_auc']:.4f}" if pd.notnull(r["roc_auc"]) else "N/A"
            formatted_stocks.append({
                "Symbol": r["symbol"],
                "N": int(r["observations"]),
                "Bullish %": f"{r['class_1_ratio']:.1%}",
                "Accuracy": f"{r['accuracy']:.1%}",
                "Bal. Acc.": f"{r['balanced_accuracy']:.1%}",
                "F1": f"{r['f1']:.4f}",
                "ROC-AUC": auc_str,
            })
        st.dataframe(pd.DataFrame(formatted_stocks), use_container_width=True, hide_index=True)
    else:
        st.warning("Per-stock evaluation not found.")

    # ── Diagnostic figures ──
    st.markdown('<div class="section-label">4 · Diagnostic Figures</div>', unsafe_allow_html=True)

    figure_tabs = {
        "ROC & PR Curves": "holdout_roc_pr_curves.png",
        "Confusion Matrix": "holdout_confusion_matrix.png",
        "Walk-Forward Folds": "walk_forward_auc.png",
        "Calibration": "probability_calibration.png",
        "Per-Stock AUC": "per_stock_auc.png",
        "Feature Importance": "feature_importance.png",
    }

    # Only create tabs for figures that exist
    available_tabs = {}
    for label, fname in figure_tabs.items():
        if (FIGURES_DIR / fname).exists():
            available_tabs[label] = fname

    if available_tabs:
        tabs = st.tabs(list(available_tabs.keys()))
        for tab, (label, fname) in zip(tabs, available_tabs.items()):
            with tab:
                st.image(str(FIGURES_DIR / fname), caption=label, use_container_width=True)
    else:
        st.info("No diagnostic figures found. Run `python run_project.py` to generate them.")

    # ── Limitations ──
    st.markdown('<div class="section-label">5 · Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - All evaluation metrics are computed on an untouched chronological holdout set.
          No information from the holdout period was used during model training.
        - The deployed Logistic Regression was selected based on walk-forward
          cross-validation (highest mean ROC-AUC across 3 expanding folds),
          prioritising generalisability and interpretability over holdout performance.
        - The best holdout ROC-AUC (Hist Gradient Boosting) marginally exceeds 0.50
          but does not demonstrate statistically significant predictive skill.
        - Results are consistent with the Efficient Market Hypothesis:
          short-term directional predictability from daily technical features alone
          is near chance level.
        """
    )


# ==============================================================================
# PAGE 4: ABOUT PROJECT
# ==============================================================================

def render_about_project():
    _page_header(
        "About This Project",
        "Academic context, methodology, feature engineering, and limitations",
        "📖",
    )

    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin-top: 0; color: #e2e8f0; font-size: 1.10rem;">🎓 Academic Context</h3>
            <p style="color: #94a3b8; font-size: 0.90rem; line-height: 1.65; margin-bottom: 0;">
                <strong style="color:#c7d2fe;">Course:</strong> B.Tech 3rd Semester — Introduction to Machine Learning (IML)<br>
                <strong style="color:#c7d2fe;">Project Title:</strong> NSE Stock Price Prediction using Machine Learning<br>
                <strong style="color:#c7d2fe;">Methodology:</strong> Point-in-time technical feature extraction,
                chronological walk-forward cross-validation, 5-day purge/embargo for forward
                return targets, and corporate-action back-adjustment using official NSE filings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">1 · Problem Formulation</div>', unsafe_allow_html=True)
    st.markdown(
        r"""
        Given an observation vector $X_t \in \mathbb{R}^{20}$ constructed strictly from
        completed market data at session $t$, predict the direction of the
        corporate-action-adjusted closing price 5 trading sessions ahead:

        $$y_t = \begin{cases} 1 & \text{if } P_{t+5} > P_t \\ 0 & \text{otherwise} \end{cases}$$

        **Leakage safeguard:** Consecutive observations share overlapping forward price windows.
        A 5-session embargo buffer is enforced between all training, validation, and holdout splits.
        """
    )

    st.markdown('<div class="section-label">2 · Dataset & Corporate Actions</div>', unsafe_allow_html=True)
    dataset = get_dataset()
    if not dataset.empty:
        n_rows = len(dataset)
        n_symbols = dataset["Symbol"].nunique()
        date_min = dataset["Date"].min().strftime("%Y-%m-%d")
        date_max = dataset["Date"].max().strftime("%Y-%m-%d")
        st.markdown(
            f"The dataset contains **{n_rows:,} daily equity records** across "
            f"**{n_symbols} NSE equities** from {date_min} to {date_max}."
        )
    else:
        st.markdown(
            f"The dataset covers **16 NSE equities** across approximately 5 years of daily data."
        )

    st.markdown(
        "Corporate actions causing mechanical price discontinuities were back-adjusted:"
    )

    ca_records = []
    for sym, details in CONFIRMED_CORPORATE_ACTIONS.items():
        ca_records.append({
            "Symbol": sym,
            "Ex-Date": details["ex_date"],
            "Action": details["action_type"],
            "Details": details["ratio_details"],
            "Factor": f"{details['adjustment_factor']:.4f}",
            "Source": details["source"],
        })
    st.dataframe(pd.DataFrame(ca_records), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">3 · Feature Engineering (20 Predictors)</div>', unsafe_allow_html=True)
    st.markdown(
        "Features are computed strictly point-in-time and grouped into 7 families:"
    )

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.markdown(
            """
            - **Momentum:** `return_1d`, `return_2d`, `return_5d`, `return_10d`, `return_20d`
            - **Trend Ratios:** `close_vs_sma_5`, `..._10`, `..._20`, `..._50`
            - **Volatility:** `volatility_10d`, `volatility_20d`
            - **Intraday:** `daily_range_pct`, `open_close_return`
            """
        )
    with feat_col2:
        st.markdown(
            """
            - **Volume:** `volume_change_1d`, `volume_vs_avg20`, `trades_vs_avg20`
            - **Delivery:** `delivery_pct`
            - **Oscillator:** `rsi_14` (Wilder RSI)
            - **Calendar:** `day_of_week`, `month`
            """
        )

    st.markdown('<div class="section-label">4 · Candidate Models</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **Dummy Baseline** — Majority-class prediction (performance floor).
        2. **Logistic Regression** — L2-regularised, balanced class weights (deployed model).
        3. **Random Forest** — 300-tree ensemble with leaf-size constraints.
        4. **Hist Gradient Boosting** — Histogram-based boosted trees.
        5. **XGBoost** — Extreme gradient boosting with depth regularisation.
        """
    )

    st.markdown('<div class="section-label">5 · Evaluation Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Walk-Forward Validation:** 3 expanding chronological folds on the development
          set (pre-holdout), each with a 5-session purge buffer.
        - **Holdout Evaluation:** Final 15% of trading dates held out and never seen
          during model selection or hyperparameter tuning.
        - **Metrics:** ROC-AUC, PR-AUC, Balanced Accuracy, Precision, Recall, F1, Brier Score.
        """
    )

    st.markdown('<div class="section-label">6 · Limitations & Ethical Considerations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="callout-warning">
            <strong>Key Limitations:</strong><br>
            <strong>No microstructure data:</strong> Daily data does not capture
            intraday order flow, bid-ask spreads, or market depth.<br>
            <strong>No fundamental/macro factors:</strong> Earnings, interest rates,
            and geopolitical events are not included in the feature set.<br>
            <strong>Transaction costs ignored:</strong> Gross directional forecasts do not
            account for STT, brokerage, GST, or market impact.<br>
            <strong>Scientific null result:</strong> Consistent with finance literature,
            short-term directional predictability from daily technical features alone
            is near chance level. The project demonstrates rigorous methodology,
            not profitable trading signals.<br><br>
            <em>All outputs are experimental statistical estimates for academic
            purposes only. They do not constitute financial advice.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main():
    selected_page = render_sidebar()

    # Strip emoji prefix for matching
    page_key = selected_page.split("  ", 1)[-1].strip() if "  " in selected_page else selected_page

    if "Market Dashboard" in page_key:
        render_dashboard()
    elif "Stock Prediction" in page_key:
        render_stock_prediction()
    elif "Model Performance" in page_key:
        render_model_performance()
    elif "About Project" in page_key:
        render_about_project()


if __name__ == "__main__":
    main()
