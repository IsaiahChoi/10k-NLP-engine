"""
feature_builder.py
==================
Build the master feature panel by combining:
  1. Text sentiment features (LM + FinBERT) for Item 1A and MD&A sections.
  2. Text statistics (word count, Fog index, vocabulary richness).
  3. Delta features (YoY / QoQ change in sentiment vs. prior filing).
  4. Equity features (abnormal returns at multiple horizons, realized vol).

Public API
----------
build_feature_panel(parsed_filings_df, prices_df, lm_dict, finbert_pipeline)
    -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_text_stats_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add text statistic columns to the filings DataFrame.

    Computes stats for both Item 1A and MD&A columns.

    Args:
        df: DataFrame with columns ``item_1a_text`` and ``mda_text``.

    Returns:
        DataFrame with additional stat columns.
    """
    from src.text_processor import compute_text_stats

    for section, col in [("1a", "item_1a_text"), ("mda", "mda_text")]:
        tqdm.pandas(desc=f"Text stats [{section}]")
        stats = df[col].fillna("").progress_apply(
            lambda t: compute_text_stats(t)
        )
        stats_df = pd.DataFrame(list(stats))
        stats_df.columns = [f"stat_{section}_{c}" for c in stats_df.columns]
        df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)
    return df


def _compute_lm_features(
    df: pd.DataFrame,
    lm_dict: dict,
) -> pd.DataFrame:
    """Add LM sentiment columns for Item 1A and MD&A.

    Args:
        df: Filings DataFrame.
        lm_dict: LM dictionary from
            :func:`sentiment_lm.load_lm_dictionary`.

    Returns:
        DataFrame with LM sentiment columns.
    """
    from src.sentiment_lm import batch_lm_sentiment

    df = batch_lm_sentiment(df, lm_dict, text_col="item_1a_text", prefix="lm_1a_")
    df = batch_lm_sentiment(df, lm_dict, text_col="mda_text", prefix="lm_mda_")
    return df


def _compute_finbert_features(
    df: pd.DataFrame,
    finbert_pipeline,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Add FinBERT sentiment columns for Item 1A and MD&A.

    Args:
        df: Filings DataFrame.
        finbert_pipeline: Loaded FinBERT pipeline.
        cache_dir: Directory for caching inference results.

    Returns:
        DataFrame with FinBERT sentiment columns.
    """
    from src.sentiment_finbert import batch_finbert_sentiment

    # Build a unique ID column for cache keys
    df = df.copy()
    df["_cache_id"] = (
        df["ticker"].astype(str)
        + "_"
        + df["filing_date"].astype(str)
        + "_"
        + df["filing_type"].astype(str)
    )

    df = batch_finbert_sentiment(
        df,
        finbert_pipeline,
        text_col="item_1a_text",
        prefix="fb_1a_",
        id_col="_cache_id",
        cache_dir=cache_dir,
    )
    df = batch_finbert_sentiment(
        df,
        finbert_pipeline,
        text_col="mda_text",
        prefix="fb_mda_",
        id_col="_cache_id",
        cache_dir=cache_dir,
    )
    df = df.drop(columns=["_cache_id"], errors="ignore")
    return df


def _compute_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute period-over-period changes in key sentiment metrics.

    For each ticker × filing_type, compute the change in a set of core
    sentiment metrics relative to the immediately preceding filing of the
    same type.

    Args:
        df: Feature DataFrame sorted by (ticker, filing_type, filing_date).

    Returns:
        DataFrame with additional ``delta_*`` columns.
    """
    df = df.sort_values(["ticker", "filing_type", "filing_date"]).copy()

    delta_cols = [
        "lm_1a_negative_pct",
        "lm_1a_positive_pct",
        "lm_1a_uncertainty_pct",
        "lm_1a_net_sentiment",
        "lm_mda_negative_pct",
        "lm_mda_net_sentiment",
        "fb_1a_finbert_net_sentiment",
        "fb_mda_finbert_net_sentiment",
        "stat_1a_word_count",
        "stat_1a_fog_index",
    ]
    # Only compute deltas for columns that actually exist
    delta_cols = [c for c in delta_cols if c in df.columns]

    grp = df.groupby(["ticker", "filing_type"])

    for col in delta_cols:
        df[f"delta_{col}"] = grp[col].diff()

    return df


# ---------------------------------------------------------------------------
# Equity feature merging
# ---------------------------------------------------------------------------

def _merge_equity_features(
    df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Merge forward returns, abnormal returns, and realized volatility.

    For each filing row, finds the next trading day after the filing date
    and looks up pre-computed equity metrics.

    Args:
        df: Filings feature DataFrame with a ``filing_date`` column.
        prices: Adjusted-close price DataFrame from
            :func:`utils.fetch_equity_data`.

    Returns:
        DataFrame with equity columns added.
    """
    from src.utils import (
        compute_abnormal_returns,
        compute_realized_vol,
        get_trading_days,
        next_trading_day,
    )

    trading_days = get_trading_days(prices)
    ab_returns = compute_abnormal_returns(prices, config.RETURN_WINDOWS)
    trail_vol = compute_realized_vol(prices, config.TRAILING_VOL_WINDOW)
    fwd_vol = compute_realized_vol(prices, config.FORWARD_VOL_WINDOW).shift(
        -config.FORWARD_VOL_WINDOW
    )

    result_rows: list[dict] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Merging equity features"):
        ticker = row["ticker"]
        fdate = pd.Timestamp(row["filing_date"]) if pd.notna(row["filing_date"]) else None

        eq_features: dict = {}

        if fdate is None or ticker not in prices.columns:
            result_rows.append(eq_features)
            continue

        # Find next trading day (event date)
        event_day = next_trading_day(fdate, trading_days)
        if event_day is None:
            result_rows.append(eq_features)
            continue

        eq_features["event_date"] = event_day

        # Abnormal returns at each window
        for w in config.RETURN_WINDOWS:
            col = f"{ticker}_abret_{w}d"
            if col in ab_returns.columns and event_day in ab_returns.index:
                val = ab_returns.loc[event_day, col]
                eq_features[f"abret_{w}d"] = val if not pd.isna(val) else np.nan
            else:
                eq_features[f"abret_{w}d"] = np.nan

        # Trailing volatility
        if ticker in trail_vol.columns and event_day in trail_vol.index:
            eq_features["trailing_vol_63d"] = trail_vol.loc[event_day, ticker]
        else:
            eq_features["trailing_vol_63d"] = np.nan

        # Forward volatility
        if ticker in fwd_vol.columns and event_day in fwd_vol.index:
            eq_features["forward_vol_63d"] = fwd_vol.loc[event_day, ticker]
        else:
            eq_features["forward_vol_63d"] = np.nan

        result_rows.append(eq_features)

    eq_df = pd.DataFrame(result_rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), eq_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_panel(
    parsed_filings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    lm_dict: dict,
    finbert_pipeline=None,
    cache_dir: Optional[Path] = None,
    run_finbert: bool = True,
) -> pd.DataFrame:
    """Build the complete feature panel combining text and equity features.

    Steps
    -----
    1. Text statistics (word count, Fog index, vocabulary richness).
    2. LM sentiment on Item 1A and MD&A.
    3. FinBERT sentiment on Item 1A and MD&A (optional, slow).
    4. Period-over-period delta features.
    5. Merge equity features (abnormal returns, volatility).
    6. Add sector labels from ``config.SECTOR_MAP``.
    7. Add fiscal quarter labels for fixed-effects use.

    Args:
        parsed_filings_df: Output of
            :func:`filing_parser.parse_all_filings`.
        prices_df: Equity price DataFrame from
            :func:`utils.fetch_equity_data`.
        lm_dict: LM dictionary from
            :func:`sentiment_lm.load_lm_dictionary`.
        finbert_pipeline: Loaded FinBERT pipeline (or ``None`` to skip
            FinBERT features).
        cache_dir: FinBERT cache directory.  Defaults to
            ``config.FINBERT_CACHE_DIR``.
        run_finbert: If ``False``, skip FinBERT inference even if
            *finbert_pipeline* is provided.

    Returns:
        Master feature panel DataFrame with one row per filing and columns
        for all text, sentiment, and equity features.
    """
    cache_dir = cache_dir or config.FINBERT_CACHE_DIR
    df = parsed_filings_df.copy()

    # Drop filings with no meaningful text
    text_mask = (
        (df["item_1a_text"].fillna("").str.len() > 50)
        | (df["mda_text"].fillna("").str.len() > 50)
    )
    n_dropped = (~text_mask).sum()
    if n_dropped > 0:
        logger.warning("Dropping %d filings with empty text sections", n_dropped)
    df = df[text_mask].reset_index(drop=True)

    logger.info("Step 1/5: Computing text statistics ...")
    df = _compute_text_stats_for_df(df)

    logger.info("Step 2/5: Computing LM sentiment features ...")
    df = _compute_lm_features(df, lm_dict)

    if run_finbert and finbert_pipeline is not None:
        logger.info("Step 3/5: Computing FinBERT sentiment features ...")
        df = _compute_finbert_features(df, finbert_pipeline, cache_dir)
    else:
        logger.info("Step 3/5: FinBERT skipped (pipeline=None or run_finbert=False)")

    logger.info("Step 4/5: Computing delta (change) features ...")
    df = _compute_delta_features(df)

    logger.info("Step 5/5: Merging equity features ...")
    df = _merge_equity_features(df, prices_df)

    # Add sector labels
    df["sector"] = df["ticker"].map(config.SECTOR_MAP).fillna("Unknown")

    # Add filing quarter (for fixed effects)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["filing_year"] = df["filing_date"].dt.year
    df["filing_quarter"] = df["filing_date"].dt.to_period("Q").astype(str)

    logger.info("Feature panel built: %d rows × %d columns", len(df), len(df.columns))
    return df


def build_feature_panel_from_disk(
    filings_dir: Optional[Path] = None,
    prices_parquet: Optional[Path] = None,
    lm_dict_path: Optional[Path] = None,
    run_finbert: bool = False,
    save_panel: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience wrapper: load all inputs from disk, build panel, optionally save.

    Intended for use in scripts and notebooks where inputs are already
    saved to disk.

    Args:
        filings_dir: Directory of downloaded filings.
        prices_parquet: Path to pre-fetched prices Parquet file.
        lm_dict_path: Path to LM master dictionary CSV.
        run_finbert: Whether to run FinBERT inference.
        save_panel: If provided, save the panel to this path (Parquet).

    Returns:
        Feature panel DataFrame.
    """
    from src.filing_parser import parse_all_filings
    from src.sentiment_lm import load_lm_dictionary
    from src.utils import fetch_equity_data, load_parquet, save_parquet

    logger.info("Parsing all filings ...")
    filings_df = parse_all_filings(filings_dir)

    logger.info("Loading equity data ...")
    if prices_parquet and Path(prices_parquet).exists():
        prices_df = load_parquet(prices_parquet)
    else:
        prices_df = fetch_equity_data()
        if prices_parquet:
            save_parquet(prices_df, prices_parquet)

    logger.info("Loading LM dictionary ...")
    lm_dict = load_lm_dictionary(lm_dict_path)

    finbert_pipeline = None
    if run_finbert:
        from src.sentiment_finbert import load_finbert
        finbert_pipeline = load_finbert()

    panel = build_feature_panel(
        filings_df,
        prices_df,
        lm_dict,
        finbert_pipeline=finbert_pipeline,
        run_finbert=run_finbert,
    )

    if save_panel:
        save_parquet(panel, save_panel)

    return panel
