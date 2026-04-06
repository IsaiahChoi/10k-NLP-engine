"""
analytics.py
============
Descriptive analytics, summary statistics, and visualization utilities for
the SEC NLP sentiment feature panel.

All plot functions return ``matplotlib.figure.Figure`` objects so they can
be embedded in Jupyter notebooks or Streamlit apps without side effects.

Public API
----------
sentiment_summary_stats(feature_panel)          -> pd.DataFrame
correlation_matrix(feature_panel, features)     -> tuple[Figure, pd.DataFrame]
sector_breakdown(feature_panel)                 -> pd.DataFrame
time_series_sentiment(feature_panel, prices_df) -> Figure
plot_event_study(event_result)                  -> Figure
plot_regression_coefs(regression_result)        -> Figure
plot_sentiment_vs_returns(feature_panel, ...)   -> Figure
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def _apply_style() -> None:
    """Apply a clean, publication-ready matplotlib style once."""
    global _STYLE_APPLIED
    if not _STYLE_APPLIED:
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "legend.fontsize": 10,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )
        _STYLE_APPLIED = True


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def sentiment_summary_stats(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for all numeric sentiment columns.

    Args:
        feature_panel: Master feature panel DataFrame.

    Returns:
        DataFrame with rows = sentiment metrics, columns = mean/median/std/
        min/max/p25/p75, rounded to 4 decimal places.
    """
    # Select numeric sentiment and text-stat columns
    sentinel_prefixes = ("lm_", "fb_", "stat_", "delta_", "abret_", "trailing_", "forward_")
    cols = [
        c for c in feature_panel.select_dtypes(include="number").columns
        if any(c.startswith(p) for p in sentinel_prefixes)
    ]

    if not cols:
        logger.warning("No sentiment columns found in feature_panel")
        return pd.DataFrame()

    stats = feature_panel[cols].agg(["mean", "median", "std", "min", "max"])
    stats.loc["p25"] = feature_panel[cols].quantile(0.25)
    stats.loc["p75"] = feature_panel[cols].quantile(0.75)
    return stats.T.round(4)


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def correlation_matrix(
    feature_panel: pd.DataFrame,
    features_list: Optional[list[str]] = None,
) -> tuple:
    """Compute and plot a correlation matrix of selected text features.

    Args:
        feature_panel: Master feature panel.
        features_list: Explicit list of column names to include.
            If None, a default set of key sentiment and return columns
            is used.

    Returns:
        Tuple of (matplotlib Figure, correlation DataFrame).
    """
    _apply_style()

    if features_list is None:
        candidates = [
            "lm_1a_negative_pct",
            "lm_1a_positive_pct",
            "lm_1a_uncertainty_pct",
            "lm_1a_net_sentiment",
            "lm_mda_negative_pct",
            "lm_mda_net_sentiment",
            "fb_1a_finbert_net_sentiment",
            "fb_mda_finbert_net_sentiment",
            "stat_1a_fog_index",
            "stat_1a_vocabulary_richness",
            "stat_1a_word_count",
            "abret_1d",
            "abret_5d",
            "abret_21d",
            "abret_63d",
            "forward_vol_63d",
        ]
        features_list = [c for c in candidates if c in feature_panel.columns]

    corr = feature_panel[features_list].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(8, len(features_list) * 0.7),
                                     max(6, len(features_list) * 0.6)))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Matrix: Sentiment Features × Equity Outcomes")
    plt.tight_layout()
    return fig, corr


# ---------------------------------------------------------------------------
# Sector breakdown
# ---------------------------------------------------------------------------

def sector_breakdown(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute average sentiment metrics grouped by sector.

    Args:
        feature_panel: Master feature panel.

    Returns:
        DataFrame with sectors as rows and average sentiment metrics as
        columns.
    """
    if "sector" not in feature_panel.columns:
        logger.warning("'sector' column not in feature_panel")
        return pd.DataFrame()

    sentiment_cols = [
        c for c in feature_panel.columns
        if any(c.startswith(p) for p in ("lm_", "fb_"))
        and "count" not in c
        and "total" not in c
        and "n_sent" not in c
    ]

    if not sentiment_cols:
        logger.warning("No sentiment columns for sector breakdown")
        return pd.DataFrame()

    return feature_panel.groupby("sector")[sentiment_cols].mean().round(4)


# ---------------------------------------------------------------------------
# Time-series sentiment
# ---------------------------------------------------------------------------

def time_series_sentiment(
    feature_panel: pd.DataFrame,
    prices_df: Optional[pd.DataFrame] = None,
    sentiment_col: str = "lm_1a_net_sentiment",
    filing_quarter_col: str = "filing_quarter",
) -> plt.Figure:
    """Plot average sentiment over time with optional market return overlay.

    Args:
        feature_panel: Master feature panel.
        prices_df: Equity price DataFrame (for SPY overlay).  Optional.
        sentiment_col: Sentiment column to aggregate.
        filing_quarter_col: Column containing filing quarter labels.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    if sentiment_col not in feature_panel.columns:
        raise ValueError(f"Column '{sentiment_col}' not in feature_panel")
    if filing_quarter_col not in feature_panel.columns:
        raise ValueError(f"Column '{filing_quarter_col}' not in feature_panel")

    ts = feature_panel.groupby(filing_quarter_col)[sentiment_col].mean().reset_index()
    ts.columns = ["quarter", "avg_sentiment"]
    ts["quarter"] = ts["quarter"].astype(str)
    ts = ts.sort_values("quarter")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    color_sent = "#2563EB"
    ax1.bar(
        range(len(ts)),
        ts["avg_sentiment"],
        color=[color_sent if v >= 0 else "#DC2626" for v in ts["avg_sentiment"]],
        alpha=0.75,
        label=sentiment_col,
    )
    ax1.set_xlabel("Filing Quarter")
    ax1.set_ylabel("Average Net Sentiment (LM)")
    ax1.set_xticks(range(len(ts)))
    ax1.set_xticklabels(ts["quarter"], rotation=75, fontsize=7)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")

    if prices_df is not None and config.SPY_TICKER in prices_df.columns:
        try:
            spy_q = (
                prices_df[config.SPY_TICKER]
                .resample("Q")
                .last()
                .pct_change()
                .dropna()
            )
            spy_q.index = spy_q.index.to_period("Q").astype(str)
            common = ts["quarter"].values
            spy_q = spy_q[spy_q.index.isin(common)]

            ax2 = ax1.twinx()
            ax2.plot(
                [list(ts["quarter"]).index(q) for q in spy_q.index if q in list(ts["quarter"])],
                spy_q.values,
                color="#16A34A",
                linewidth=1.8,
                label="SPY Quarterly Return",
                marker="o",
                markersize=3,
            )
            ax2.set_ylabel("SPY Quarterly Return")
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
            ax2.legend(loc="upper right")
        except Exception as exc:
            logger.warning("Could not overlay SPY returns: %s", exc)

    ax1.legend(loc="upper left")
    ax1.set_title("Average Sentiment Over Time (Filing Quarter)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Event study plot
# ---------------------------------------------------------------------------

def plot_event_study(
    event_result: dict,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of quantile average returns for an event study result.

    Args:
        event_result: Output dict from
            :func:`return_linker.event_study`.
        title: Plot title.  If None, auto-generated from result metadata.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    means = event_result.get("quantile_means", [])
    counts = event_result.get("quantile_counts", [])
    spread = event_result.get("spread", np.nan)
    tstat = event_result.get("t_stat", np.nan)
    pval = event_result.get("p_value", np.nan)
    sent_col = event_result.get("sentiment_column", "Sentiment")
    ret_col = event_result.get("return_column", "Return")
    n_q = len(means)

    if not means:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"Q{i+1}\n(n={counts[i]})" for i in range(n_q)] if counts else [f"Q{i+1}" for i in range(n_q)]
    colors = ["#DC2626"] + ["#94A3B8"] * (n_q - 2) + ["#16A34A"] if n_q >= 3 else ["#2563EB"] * n_q
    bars = ax.bar(labels, [m * 100 for m in means], color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(f"Average {ret_col} (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    significance = ""
    if not pd.isna(pval):
        if pval < 0.01:
            significance = " ***"
        elif pval < 0.05:
            significance = " **"
        elif pval < 0.10:
            significance = " *"

    _title = title or (
        f"Event Study: {sent_col} → {ret_col}\n"
        f"Spread = {spread*100:.2f}%,  t = {tstat:.2f}{significance}"
    )
    ax.set_title(_title)
    ax.set_xlabel(f"Quantile of {sent_col} (Low → High)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regression coefficient plot
# ---------------------------------------------------------------------------

def plot_regression_coefs(
    regression_result,
    title: str = "OLS Coefficient Estimates (HAC SEs)",
    n_top: int = 15,
) -> plt.Figure:
    """Horizontal bar chart of regression coefficients with confidence intervals.

    Args:
        regression_result: statsmodels ``RegressionResultsWrapper``.
        title: Plot title.
        n_top: Maximum number of coefficients to display (excluding
            constant and quarter FE dummies).

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    params = regression_result.params
    conf = regression_result.conf_int()
    pvals = regression_result.pvalues

    # Filter out intercept and quarter fixed-effect dummies
    mask = ~(params.index.str.startswith("q_") | (params.index == "const"))
    params = params[mask]
    conf = conf[mask]
    pvals = pvals[mask]

    # Sort by absolute magnitude, take top n
    order = params.abs().nlargest(n_top).index
    params = params[order]
    conf = conf.loc[order]
    pvals = pvals.loc[order]

    fig, ax = plt.subplots(figsize=(8, max(5, len(params) * 0.4)))
    colors = ["#DC2626" if p < 0 else "#2563EB" for p in params]
    ax.barh(range(len(params)), params.values, color=colors, alpha=0.75, height=0.6)
    ax.errorbar(
        x=params.values,
        y=range(len(params)),
        xerr=[
            params.values - conf.iloc[:, 0].values,
            conf.iloc[:, 1].values - params.values,
        ],
        fmt="none",
        color="black",
        linewidth=1.5,
        capsize=4,
    )
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sentiment vs. returns scatter
# ---------------------------------------------------------------------------

def plot_sentiment_vs_returns(
    feature_panel: pd.DataFrame,
    sentiment_col: str = "lm_1a_negative_pct",
    return_col: str = "abret_21d",
    hue_col: str = "sector",
    title: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of sentiment vs. forward abnormal returns.

    Args:
        feature_panel: Master feature panel.
        sentiment_col: X-axis sentiment feature.
        return_col: Y-axis return outcome.
        hue_col: Column for color-coding points (e.g. sector).
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    needed = [sentiment_col, return_col]
    missing = [c for c in needed if c not in feature_panel.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    plot_df = feature_panel[[sentiment_col, return_col]
                             + ([hue_col] if hue_col in feature_panel.columns else [])].dropna()

    fig, ax = plt.subplots(figsize=(9, 6))

    if hue_col in plot_df.columns:
        sectors = plot_df[hue_col].unique()
        palette = sns.color_palette("tab10", len(sectors))
        for sector, color in zip(sectors, palette):
            sub = plot_df[plot_df[hue_col] == sector]
            ax.scatter(
                sub[sentiment_col],
                sub[return_col],
                label=sector,
                color=color,
                alpha=0.5,
                s=25,
                edgecolors="none",
            )
        ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    else:
        ax.scatter(plot_df[sentiment_col], plot_df[return_col], alpha=0.4, s=25)

    # OLS trend line
    x = plot_df[sentiment_col].values
    y = plot_df[return_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() > 10:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, p(x_line), "k--", linewidth=1.5, label="OLS trend")

    ax.set_xlabel(sentiment_col)
    ax.set_ylabel(return_col)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_title(title or f"{sentiment_col} vs. {return_col}")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Word-count distribution
# ---------------------------------------------------------------------------

def plot_word_count_distribution(
    feature_panel: pd.DataFrame,
    col: str = "stat_1a_word_count",
    title: str = "Distribution of Item 1A Word Count",
) -> plt.Figure:
    """Histogram of word counts with KDE overlay.

    Args:
        feature_panel: Master feature panel.
        col: Word count column name.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    if col not in feature_panel.columns:
        raise ValueError(f"Column '{col}' not in feature_panel")

    data = feature_panel[col].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data, kde=True, bins=40, color="#2563EB", alpha=0.65, ax=ax)
    ax.axvline(data.median(), color="#DC2626", linewidth=1.8, linestyle="--",
               label=f"Median: {data.median():,.0f}")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
