"""
streamlit_app.py
================
Interactive Streamlit dashboard for the 10k-nlp-sentiment-engine.

Features
--------
- Dropdown to select a ticker and filing type.
- Filing history table: date, type, word count, LM sentiment, FinBERT sentiment.
- Time-series chart of LM net sentiment vs. adjusted close price.
- Raw text viewer for Item 1A (Risk Factors) of any selected filing.
- Sector comparison bar chart.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure parent directory is on path for config imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="10-K NLP Sentiment Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

PANEL_PATHS = [
    ROOT_DIR / "data" / "feature_panel.parquet",
    ROOT_DIR / "data" / "feature_panel.csv",
]

PRICES_PATHS = [
    ROOT_DIR / "data" / "prices.parquet",
    ROOT_DIR / "data" / "prices.csv",
]


@st.cache_data(show_spinner="Loading feature panel ...")
def load_panel() -> pd.DataFrame:
    """Load the pre-built feature panel from disk.

    Returns:
        Feature panel DataFrame, or a demo DataFrame if no data exists.
    """
    for p in PANEL_PATHS:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            else:
                return pd.read_csv(p, parse_dates=["filing_date"])

    # Generate synthetic demo data if no real panel exists
    st.warning(
        "No pre-built feature panel found. Showing synthetic demo data.\n"
        "Run `build_feature_panel_from_disk()` to generate real data."
    )
    return _generate_demo_panel()


@st.cache_data(show_spinner="Loading equity prices ...")
def load_prices() -> pd.DataFrame:
    """Load equity prices from disk.

    Returns:
        Price DataFrame or empty DataFrame if not found.
    """
    for p in PRICES_PATHS:
        if p.exists():
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df
    return pd.DataFrame()


def _generate_demo_panel() -> pd.DataFrame:
    """Generate a small synthetic panel for demonstration purposes.

    Returns:
        Synthetic DataFrame mimicking the real feature panel schema.
    """
    rng = np.random.default_rng(42)
    tickers = config.TICKERS[:10]
    years = range(2018, 2024)
    rows = []
    for ticker in tickers:
        for year in years:
            for qtr in [1, 4]:
                month = 3 if qtr == 1 else 12
                rows.append(
                    {
                        "ticker": ticker,
                        "filing_type": "10-K" if qtr == 4 else "10-Q",
                        "filing_date": pd.Timestamp(f"{year}-{month:02d}-15"),
                        "filing_quarter": f"{year}Q{qtr}",
                        "sector": config.SECTOR_MAP.get(ticker, "Unknown"),
                        "stat_1a_word_count": int(rng.integers(3000, 15000)),
                        "stat_mda_word_count": int(rng.integers(5000, 25000)),
                        "stat_1a_fog_index": round(float(rng.uniform(14, 24)), 2),
                        "stat_1a_vocabulary_richness": round(float(rng.uniform(0.3, 0.7)), 4),
                        "lm_1a_negative_pct": round(float(rng.uniform(0.02, 0.12)), 4),
                        "lm_1a_positive_pct": round(float(rng.uniform(0.005, 0.04)), 4),
                        "lm_1a_uncertainty_pct": round(float(rng.uniform(0.01, 0.06)), 4),
                        "lm_1a_net_sentiment": round(float(rng.uniform(-0.1, 0.02)), 4),
                        "lm_mda_negative_pct": round(float(rng.uniform(0.01, 0.08)), 4),
                        "lm_mda_net_sentiment": round(float(rng.uniform(-0.07, 0.02)), 4),
                        "fb_1a_finbert_net_sentiment": round(float(rng.uniform(-0.6, 0.2)), 4),
                        "fb_mda_finbert_net_sentiment": round(float(rng.uniform(-0.5, 0.3)), 4),
                        "fb_1a_finbert_negative_pct": round(float(rng.uniform(0.2, 0.7)), 4),
                        "fb_1a_finbert_positive_pct": round(float(rng.uniform(0.05, 0.3)), 4),
                        "abret_1d": round(float(rng.normal(0, 0.015)), 4),
                        "abret_5d": round(float(rng.normal(0, 0.03)), 4),
                        "abret_21d": round(float(rng.normal(0, 0.06)), 4),
                        "abret_63d": round(float(rng.normal(0, 0.10)), 4),
                        "trailing_vol_63d": round(float(rng.uniform(0.15, 0.55)), 4),
                        "forward_vol_63d": round(float(rng.uniform(0.15, 0.55)), 4),
                        "item_1a_text": (
                            f"[Demo] {ticker} faces various risks in {year} including "
                            "macroeconomic uncertainty, regulatory changes, and competitive pressures. "
                            "The company may experience significant losses if these risks materialize. "
                            "Litigation risks remain elevated. The company cannot guarantee future profitability."
                        ),
                        "mda_text": (
                            f"[Demo] Management's Discussion for {ticker} FY{year}: "
                            "Revenue grew 12% year-over-year, driven by strong product demand. "
                            "Operating margins expanded despite cost headwinds. "
                            "The company remains cautiously optimistic about the outlook."
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

def render_sidebar(panel: pd.DataFrame) -> dict:
    """Render sidebar controls and return selected filter values.

    Args:
        panel: Feature panel DataFrame.

    Returns:
        Dict with selected ticker, filing_type, and year_range.
    """
    st.sidebar.header("Filters")

    tickers_available = sorted(panel["ticker"].dropna().unique())
    selected_ticker = st.sidebar.selectbox(
        "Ticker", tickers_available, index=0
    )

    filing_types = sorted(panel["filing_type"].dropna().unique())
    selected_type = st.sidebar.multiselect(
        "Filing Type", filing_types, default=filing_types
    )

    min_year = int(panel["filing_date"].dt.year.min())
    max_year = int(panel["filing_date"].dt.year.max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**10-K NLP Sentiment Engine**\n\n"
        "Powered by Loughran-McDonald dictionary & FinBERT.\n\n"
        "For research purposes only."
    )

    return {
        "ticker": selected_ticker,
        "filing_types": selected_type,
        "year_range": year_range,
    }


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Render the dashboard header and description."""
    st.title("📊 10-K NLP Sentiment Engine")
    st.markdown(
        "End-to-end NLP pipeline for SEC 10-K / 10-Q sentiment analysis "
        "linked to subsequent equity returns and volatility."
    )
    st.divider()


def render_filing_history(ticker_df: pd.DataFrame) -> None:
    """Display a formatted filing history table for the selected ticker.

    Args:
        ticker_df: Filtered DataFrame for a single ticker.
    """
    st.subheader("📁 Filing History")

    display_cols = {
        "filing_date": "Filing Date",
        "filing_type": "Type",
        "stat_1a_word_count": "Item 1A Words",
        "stat_1a_fog_index": "Fog Index",
        "lm_1a_net_sentiment": "LM Net Sent.",
        "lm_1a_negative_pct": "LM Neg. %",
        "lm_1a_uncertainty_pct": "LM Uncert. %",
        "fb_1a_finbert_net_sentiment": "FinBERT Net Sent.",
        "fb_1a_finbert_negative_pct": "FinBERT Neg. %",
        "abret_1d": "1-Day Abret",
        "abret_21d": "21-Day Abret",
    }

    existing = {k: v for k, v in display_cols.items() if k in ticker_df.columns}
    display_df = ticker_df[list(existing.keys())].copy()
    display_df.columns = list(existing.values())

    # Format numeric columns
    pct_cols = [
        "LM Net Sent.", "LM Neg. %", "LM Uncert. %",
        "FinBERT Net Sent.", "FinBERT Neg. %",
        "1-Day Abret", "21-Day Abret"
    ]
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )

    if "Filing Date" in display_df.columns:
        display_df["Filing Date"] = pd.to_datetime(
            display_df["Filing Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df.sort_values("Filing Date", ascending=False)
        if "Filing Date" in display_df.columns
        else display_df,
        use_container_width=True,
        height=300,
    )


def render_sentiment_vs_price(
    ticker_df: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
) -> None:
    """Plot LM net sentiment alongside the adjusted-close price.

    Args:
        ticker_df: Filtered DataFrame for a single ticker.
        prices: Equity price DataFrame.
        ticker: Ticker symbol string.
    """
    st.subheader("📈 Sentiment vs. Stock Price")

    sent_col = "lm_1a_net_sentiment"
    if sent_col not in ticker_df.columns or ticker_df[sent_col].isna().all():
        st.info("No LM sentiment data available for this ticker.")
        return

    ts = ticker_df[["filing_date", sent_col]].dropna().sort_values("filing_date")

    fig = go.Figure()

    # Sentiment bars
    fig.add_trace(
        go.Bar(
            x=ts["filing_date"],
            y=ts[sent_col],
            name="LM Net Sentiment",
            marker_color=[
                "#16A34A" if v >= 0 else "#DC2626" for v in ts[sent_col]
            ],
            opacity=0.7,
            yaxis="y1",
        )
    )

    # Price line overlay
    if not prices.empty and ticker in prices.columns:
        price_ts = prices[ticker].dropna()
        fig.add_trace(
            go.Scatter(
                x=price_ts.index,
                y=price_ts.values,
                name=f"{ticker} Price",
                line=dict(color="#1D4ED8", width=2),
                yaxis="y2",
            )
        )

    fig.update_layout(
        yaxis=dict(title="LM Net Sentiment", side="left"),
        yaxis2=dict(title="Adjusted Close ($)", overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h"),
        hovermode="x unified",
        height=420,
        margin=dict(t=30, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_text_viewer(ticker_df: pd.DataFrame) -> None:
    """Allow the user to select a filing and read the Item 1A text.

    Args:
        ticker_df: Filtered DataFrame for a single ticker.
    """
    st.subheader("📄 Item 1A Risk Factors Text Viewer")

    if "item_1a_text" not in ticker_df.columns:
        st.info("No Item 1A text available in the dataset.")
        return

    valid = ticker_df[ticker_df["item_1a_text"].fillna("").str.len() > 50].copy()
    if valid.empty:
        st.info("No Item 1A text extracted for this ticker/filter combination.")
        return

    valid["label"] = (
        valid["filing_date"].dt.strftime("%Y-%m-%d").fillna("?")
        + " | "
        + valid["filing_type"].fillna("?")
    )
    options = valid["label"].tolist()
    selected_label = st.selectbox("Select Filing", options, index=0)
    row = valid[valid["label"] == selected_label].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Word Count", f"{row.get('stat_1a_word_count', 'N/A'):,}")
    col2.metric("LM Net Sentiment", f"{row.get('lm_1a_net_sentiment', 0):.3f}")
    col3.metric("FinBERT Net Sent.", f"{row.get('fb_1a_finbert_net_sentiment', 0):.3f}")
    col4.metric("Fog Index", f"{row.get('stat_1a_fog_index', 0):.1f}")

    with st.expander("📖 Item 1A Text (Risk Factors)", expanded=True):
        text = row.get("item_1a_text", "")
        # Show first 3000 chars to avoid huge renders
        display_text = text[:3000] + ("…[truncated]" if len(text) > 3000 else "")
        st.markdown(
            f"<div style='font-size:13px;line-height:1.6;font-family:Georgia,serif;"
            f"background:#F8FAFC;padding:16px;border-radius:6px;border:1px solid #E2E8F0'>"
            f"{display_text.replace(chr(10), '<br>')}</div>",
            unsafe_allow_html=True,
        )


def render_sector_comparison(panel: pd.DataFrame) -> None:
    """Bar chart of average LM negativity by sector.

    Args:
        panel: Full (unfiltered) feature panel.
    """
    st.subheader("🏭 Sector Sentiment Comparison")

    neg_col = "lm_1a_negative_pct"
    uncert_col = "lm_1a_uncertainty_pct"

    if "sector" not in panel.columns or neg_col not in panel.columns:
        st.info("Sector or sentiment data not available.")
        return

    sector_df = (
        panel.groupby("sector")[[neg_col, uncert_col]]
        .mean()
        .reset_index()
        .sort_values(neg_col, ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="LM Negativity %",
            x=sector_df[neg_col] * 100,
            y=sector_df["sector"],
            orientation="h",
            marker_color="#DC2626",
        )
    )
    fig.add_trace(
        go.Bar(
            name="LM Uncertainty %",
            x=sector_df[uncert_col] * 100,
            y=sector_df["sector"],
            orientation="h",
            marker_color="#F59E0B",
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Average % of Words",
        yaxis_title="",
        height=380,
        margin=dict(t=20, b=30, l=120),
        legend=dict(x=0.7, y=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_overview(ticker_df: pd.DataFrame, ticker: str) -> None:
    """Show a row of headline KPI metrics for the selected ticker.

    Args:
        ticker_df: Filtered DataFrame.
        ticker: Ticker symbol.
    """
    st.subheader(f"🔑 {ticker} — Key Metrics Summary")
    n_filings = len(ticker_df)
    avg_neg = ticker_df.get("lm_1a_negative_pct", pd.Series([np.nan])).mean()
    avg_finbert = ticker_df.get("fb_1a_finbert_net_sentiment", pd.Series([np.nan])).mean()
    avg_abret = ticker_df.get("abret_21d", pd.Series([np.nan])).mean()
    avg_words = ticker_df.get("stat_1a_word_count", pd.Series([np.nan])).mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filings", n_filings)
    c2.metric("Avg Item 1A Words", f"{avg_words:,.0f}" if pd.notna(avg_words) else "—")
    c3.metric("Avg LM Negativity", f"{avg_neg:.2%}" if pd.notna(avg_neg) else "—")
    c4.metric("Avg FinBERT Net Sent.", f"{avg_finbert:.3f}" if pd.notna(avg_finbert) else "—")
    c5.metric("Avg 21d Abret", f"{avg_abret:.2%}" if pd.notna(avg_abret) else "—")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit dashboard."""
    render_header()

    panel = load_panel()
    prices = load_prices()

    filters = render_sidebar(panel)

    # Apply filters
    ticker = filters["ticker"]
    year_min, year_max = filters["year_range"]

    filtered_panel = panel[
        (panel["ticker"] == ticker)
        & (panel["filing_type"].isin(filters["filing_types"] or panel["filing_type"].unique()))
        & (panel["filing_date"].dt.year >= year_min)
        & (panel["filing_date"].dt.year <= year_max)
    ].copy()

    if filtered_panel.empty:
        st.warning("No filings match the selected filters.")
        return

    render_metrics_overview(filtered_panel, ticker)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📁 Filing History", "📈 Sentiment & Price", "📄 Risk Factors", "🏭 Sector View"]
    )

    with tab1:
        render_filing_history(filtered_panel)

    with tab2:
        render_sentiment_vs_price(filtered_panel, prices, ticker)

    with tab3:
        render_text_viewer(filtered_panel)

    with tab4:
        render_sector_comparison(panel)


if __name__ == "__main__":
    main()
