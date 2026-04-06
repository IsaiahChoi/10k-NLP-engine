"""
utils.py
========
Shared utility functions: logging setup, file I/O helpers, date helpers,
equity data fetching via yfinance.

Public API
----------
setup_logging(level)
next_trading_day(date, trading_days)
compute_returns(prices, windows)
fetch_equity_data(tickers, start, end)              -> pd.DataFrame
compute_realized_vol(prices, window)                -> pd.Series
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """Configure root-level logging with a consistent format.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
        log_file: Optional path to also write logs to a file.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Equity data via yfinance
# ---------------------------------------------------------------------------

def fetch_equity_data(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    include_spy: bool = True,
) -> pd.DataFrame:
    """Download daily adjusted-close prices for a list of tickers.

    Uses yfinance with a 2-second retry on failure.

    Args:
        tickers: List of ticker symbols.  Defaults to ``config.TICKERS``.
        start: Start date "YYYY-MM-DD".  Defaults to ``config.DATE_RANGE[0]``.
        end: End date "YYYY-MM-DD".  Defaults to ``config.DATE_RANGE[1]``.
        include_spy: If True, always include SPY (S&P 500 ETF) for
            computing abnormal returns.

    Returns:
        DataFrame indexed by date (DatetimeIndex) with one column per ticker
        containing adjusted-close prices.  Missing values are forward-filled
        then backward-filled to handle non-trading days.
    """
    import yfinance as yf

    tickers = list(tickers or config.TICKERS)
    if include_spy and config.SPY_TICKER not in tickers:
        tickers = tickers + [config.SPY_TICKER]

    start = start or config.DATE_RANGE[0]
    end = end or config.DATE_RANGE[1]

    logger.info("Fetching equity prices for %d tickers (%s to %s)", len(tickers), start, end)

    for attempt in range(3):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            break
        except Exception as exc:
            logger.warning("yfinance attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    else:
        logger.error("yfinance download failed after 3 attempts")
        return pd.DataFrame()

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices.ffill().bfill()
    logger.info("Equity data shape: %s", prices.shape)
    return prices


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_returns(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute log returns over multiple forward windows.

    Args:
        prices: DataFrame of adjusted-close prices (dates as index, tickers
            as columns).
        windows: List of forward window lengths in trading days.
            Defaults to ``config.RETURN_WINDOWS``.

    Returns:
        DataFrame with MultiIndex columns (ticker, window_label), containing
        cumulative log returns for each window starting from each date.
    """
    windows = windows or config.RETURN_WINDOWS
    log_ret = np.log(prices / prices.shift(1))

    result_frames: list[pd.DataFrame] = []
    for w in windows:
        # Forward cumulative return: sum of next w days' log returns
        fwd = log_ret.shift(-1).rolling(window=w, min_periods=w).sum().shift(-(w - 1))
        fwd.columns = pd.MultiIndex.from_tuples(
            [(col, f"ret_{w}d") for col in fwd.columns]
        )
        result_frames.append(fwd)

    return pd.concat(result_frames, axis=1)


def compute_abnormal_returns(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
    benchmark: str = config.SPY_TICKER,
) -> pd.DataFrame:
    """Compute abnormal returns (ticker minus benchmark) for each window.

    Args:
        prices: Adjusted-close price DataFrame (tickers as columns).
        windows: Forward windows in trading days.
        benchmark: Benchmark ticker column name (default "SPY").

    Returns:
        DataFrame with columns like "AAPL_abret_5d", "MSFT_abret_5d", etc.
    """
    windows = windows or config.RETURN_WINDOWS
    log_ret = np.log(prices / prices.shift(1))

    if benchmark not in log_ret.columns:
        logger.warning("Benchmark %s not in prices — abnormal returns = raw", benchmark)
        bm_ret = pd.Series(0.0, index=log_ret.index)
    else:
        bm_ret = log_ret[benchmark]

    non_bm = [c for c in log_ret.columns if c != benchmark]
    excess = log_ret[non_bm].subtract(bm_ret, axis=0)

    result_frames: list[pd.DataFrame] = []
    for w in windows:
        fwd_excess = excess.shift(-1).rolling(window=w, min_periods=w).sum().shift(-(w - 1))
        fwd_excess.columns = [f"{col}_abret_{w}d" for col in fwd_excess.columns]
        result_frames.append(fwd_excess)

    return pd.concat(result_frames, axis=1)


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

def compute_realized_vol(
    prices: pd.DataFrame,
    window: int = config.TRAILING_VOL_WINDOW,
    annualize: bool = True,
) -> pd.DataFrame:
    """Compute rolling realized volatility from log returns.

    Args:
        prices: Adjusted-close price DataFrame.
        window: Rolling window in trading days.
        annualize: If True, multiply by sqrt(252) to annualize.

    Returns:
        DataFrame of rolling realized vol, same shape as *prices*.
    """
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window=window, min_periods=window // 2).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv


# ---------------------------------------------------------------------------
# Trading calendar helpers
# ---------------------------------------------------------------------------

def get_trading_days(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the index of *prices* as a DatetimeIndex of trading days.

    Args:
        prices: Price DataFrame with DatetimeIndex.

    Returns:
        Sorted DatetimeIndex.
    """
    return prices.index.sort_values()


def next_trading_day(
    dt: Union[str, pd.Timestamp, date],
    trading_days: pd.DatetimeIndex,
) -> Optional[pd.Timestamp]:
    """Return the first trading day on or after *dt*.

    Args:
        dt: Target date (string "YYYY-MM-DD", Timestamp, or date).
        trading_days: Sorted DatetimeIndex of all valid trading days.

    Returns:
        ``pd.Timestamp`` of the next (or same-day) trading day, or
        ``None`` if *dt* is after the last available trading day.
    """
    dt = pd.Timestamp(dt)
    future = trading_days[trading_days >= dt]
    if future.empty:
        return None
    return future[0]


# ---------------------------------------------------------------------------
# DataFrame I/O helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    """Save a DataFrame to Parquet format.

    Args:
        df: DataFrame to save.
        path: Destination file path (will add .parquet extension if missing).
        **kwargs: Additional arguments forwarded to ``df.to_parquet``.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True, **kwargs)
    logger.info("Saved %s rows to %s", len(df), path)


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a DataFrame from Parquet format.

    Args:
        path: Source file path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists() and not path.suffix:
        path = path.with_suffix(".parquet")
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)
