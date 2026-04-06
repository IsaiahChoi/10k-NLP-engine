"""
return_linker.py
================
Statistical linkage between text features and subsequent equity returns /
volatility.  Implements:

  1. Quantile-based event studies with Newey-West t-tests.
  2. Pooled OLS cross-sectional regressions with HAC standard errors.
  3. Fama-MacBeth (1973) two-pass regressions with Newey-West correction.

Public API
----------
event_study(feature_panel, sentiment_column, return_column, n_quantiles)
    -> dict

cross_sectional_regression(feature_panel, y_column, x_columns, controls,
                           add_quarter_fe)
    -> RegressionResultsWrapper

fama_macbeth_regression(feature_panel, y_column, x_columns)
    -> pd.DataFrame
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Newey-West t-statistic helper
# ---------------------------------------------------------------------------

def _newey_west_tstat(series: pd.Series, lags: int = config.NW_LAGS) -> tuple[float, float]:
    """Compute t-statistic for H₀: mean = 0 with Newey-West standard error.

    Args:
        series: Time series of values (e.g. quarterly coefficients).
        lags: Number of Newey-West lags.

    Returns:
        Tuple of (t-statistic, two-sided p-value).
    """
    series = series.dropna()
    n = len(series)
    if n < 3:
        return (np.nan, np.nan)

    mean_val = series.mean()

    # OLS with constant only → use statsmodels HAC SE
    y = series.values.reshape(-1, 1)
    x = np.ones((n, 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y.flatten(), x)
            res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=True)
        tstat = res.tvalues[0]
        pval = res.pvalues[0]
    except Exception:
        # Fallback: simple t-test
        se = series.std() / np.sqrt(n)
        tstat = mean_val / se if se > 0 else 0.0
        pval = 2 * (1 - stats.t.cdf(abs(tstat), df=n - 1))

    return (float(tstat), float(pval))


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------

def event_study(
    feature_panel: pd.DataFrame,
    sentiment_column: str,
    return_column: str,
    n_quantiles: int = config.N_QUANTILES,
) -> dict:
    """Quantile-based event study linking sentiment to forward returns.

    Sorts filings into *n_quantiles* groups by *sentiment_column*, then
    computes the average *return_column* in each group and tests whether
    the top-minus-bottom spread is statistically significant.

    Args:
        feature_panel: Master feature panel DataFrame.
        sentiment_column: Column name of the sentiment signal
            (e.g. "lm_1a_negative_pct").
        return_column: Column name of the return outcome
            (e.g. "abret_21d").
        n_quantiles: Number of quantile groups (default: 3 → terciles).

    Returns:
        Dict with keys:
            - ``quantile_means``   : list of mean returns per quantile (low→high)
            - ``quantile_counts``  : list of filing counts per quantile
            - ``spread``           : top - bottom quantile mean return
            - ``t_stat``           : Newey-West t-statistic for the spread
            - ``p_value``          : two-sided p-value
            - ``sentiment_column`` : input column name
            - ``return_column``    : input return column name
            - ``n_quantiles``      : number of quantile groups
    """
    required = [sentiment_column, return_column]
    for col in required:
        if col not in feature_panel.columns:
            raise ValueError(f"Column '{col}' not in feature_panel")

    df = feature_panel[[sentiment_column, return_column]].dropna()
    if len(df) < n_quantiles * 5:
        logger.warning("Too few observations (%d) for %d quantiles", len(df), n_quantiles)
        return {
            "quantile_means": [],
            "quantile_counts": [],
            "spread": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "sentiment_column": sentiment_column,
            "return_column": return_column,
            "n_quantiles": n_quantiles,
        }

    labels = list(range(1, n_quantiles + 1))
    df = df.copy()
    df["quantile"] = pd.qcut(df[sentiment_column], q=n_quantiles, labels=labels, duplicates="drop")

    grp = df.groupby("quantile")[return_column]
    quantile_means = grp.mean().tolist()
    quantile_counts = grp.count().tolist()

    if len(quantile_means) < 2:
        spread = np.nan
        tstat, pval = np.nan, np.nan
    else:
        spread = quantile_means[-1] - quantile_means[0]

        # Construct time series of spread for NW t-test
        # (use quarterly average of [top-quantile ret] - [bottom-quantile ret])
        if "filing_quarter" in feature_panel.columns:
            df_full = feature_panel[[sentiment_column, return_column, "filing_quarter"]].dropna()
            df_full = df_full.copy()
            df_full["quantile"] = pd.qcut(
                df_full[sentiment_column],
                q=n_quantiles,
                labels=labels,
                duplicates="drop",
            )
            top = df_full[df_full["quantile"] == n_quantiles].groupby("filing_quarter")[return_column].mean()
            bot = df_full[df_full["quantile"] == 1].groupby("filing_quarter")[return_column].mean()
            spread_ts = (top - bot).dropna()
            tstat, pval = _newey_west_tstat(spread_ts)
        else:
            # Simple two-sample t-test fallback
            top_vals = df[df["quantile"] == n_quantiles][return_column]
            bot_vals = df[df["quantile"] == 1][return_column]
            tstat, pval = stats.ttest_ind(top_vals, bot_vals, equal_var=False)

    return {
        "quantile_means": quantile_means,
        "quantile_counts": quantile_counts,
        "spread": float(spread) if not pd.isna(spread) else np.nan,
        "t_stat": float(tstat) if not pd.isna(tstat) else np.nan,
        "p_value": float(pval) if not pd.isna(pval) else np.nan,
        "sentiment_column": sentiment_column,
        "return_column": return_column,
        "n_quantiles": n_quantiles,
    }


# ---------------------------------------------------------------------------
# Cross-sectional OLS regression
# ---------------------------------------------------------------------------

def cross_sectional_regression(
    feature_panel: pd.DataFrame,
    y_column: str,
    x_columns: list[str],
    controls: Optional[list[str]] = None,
    add_quarter_fe: bool = True,
    nw_lags: int = config.NW_LAGS,
):
    """Pooled OLS regression with optional quarter fixed effects and HAC SEs.

    Model: y = α + β·X + γ·controls + δ·quarter_FE + ε

    Fixed effects are implemented via dummy variables (within-transformation
    is numerically identical for balanced panels, and this approach works
    for unbalanced panels too).

    Args:
        feature_panel: Master feature panel DataFrame.
        y_column: Name of the dependent variable column.
        x_columns: List of primary regressor column names.
        controls: Additional control variable column names.
        add_quarter_fe: If True, add filing-quarter dummy variables.
        nw_lags: Number of lags for Newey-West HAC standard errors.

    Returns:
        statsmodels ``RegressionResultsWrapper`` with HAC standard errors.

    Raises:
        ValueError: If required columns are missing.
    """
    controls = controls or []
    all_x = x_columns + controls

    needed = [y_column] + all_x
    missing = [c for c in needed if c not in feature_panel.columns]
    if missing:
        raise ValueError(f"Columns not found in feature_panel: {missing}")

    df = feature_panel[[y_column] + all_x + (["filing_quarter"] if add_quarter_fe else [])].dropna(subset=[y_column] + all_x)
    df = df.copy()

    if len(df) < len(all_x) + 10:
        raise ValueError(f"Too few observations ({len(df)}) for regression with {len(all_x)} regressors")

    # Standardize continuous regressors (mean 0, std 1) for interpretability
    df[all_x] = df[all_x].apply(lambda col: (col - col.mean()) / (col.std() + 1e-12))

    y = df[y_column].values
    X = df[all_x].copy()

    if add_quarter_fe and "filing_quarter" in df.columns:
        quarter_dummies = pd.get_dummies(df["filing_quarter"], prefix="q", drop_first=True)
        X = pd.concat([X.reset_index(drop=True), quarter_dummies.reset_index(drop=True)], axis=1)

    X = sm.add_constant(X, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X.astype(float))
        results = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": nw_lags},
            use_t=True,
        )

    logger.info(
        "OLS [%s ~ %s] n=%d  R²=%.4f",
        y_column, "+".join(x_columns), len(df), results.rsquared,
    )
    return results


# ---------------------------------------------------------------------------
# Fama-MacBeth regression
# ---------------------------------------------------------------------------

def fama_macbeth_regression(
    feature_panel: pd.DataFrame,
    y_column: str,
    x_columns: list[str],
    time_col: str = "filing_quarter",
    nw_lags: int = config.NW_LAGS,
    min_obs_per_period: int = 10,
) -> pd.DataFrame:
    """Fama-MacBeth (1973) two-pass regression.

    **Pass 1**: For each time period (filing quarter), run a cross-sectional
    OLS of y on X.  Collect the slope vector β_t.

    **Pass 2**: Average β_t across periods.  Compute Newey-West t-statistics
    on the time series of β_t.

    Args:
        feature_panel: Master feature panel DataFrame.
        y_column: Dependent variable column.
        x_columns: Regressor column names.
        time_col: Column defining time periods
            (default ``"filing_quarter"``).
        nw_lags: Newey-West lag for Pass 2.
        min_obs_per_period: Minimum cross-sectional observations required
            to include a period in Pass 2.

    Returns:
        DataFrame with columns:
            variable, mean_coef, std_coef, t_stat, p_value, n_periods.
    """
    needed = [y_column, time_col] + x_columns
    missing = [c for c in needed if c not in feature_panel.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    df = feature_panel[[y_column, time_col] + x_columns].dropna()
    periods = sorted(df[time_col].unique())

    period_coefs: dict[str, list[float]] = {col: [] for col in x_columns}
    valid_periods: list[str] = []

    for period in periods:
        pdata = df[df[time_col] == period]
        if len(pdata) < min_obs_per_period:
            continue

        y = pdata[y_column].values
        X_raw = pdata[x_columns].values
        # Standardize within period
        col_std = X_raw.std(axis=0)
        col_std[col_std < 1e-12] = 1.0
        X_std = (X_raw - X_raw.mean(axis=0)) / col_std
        X = sm.add_constant(X_std, has_constant="add")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(y, X).fit()
            for i, col in enumerate(x_columns):
                # Coefficients start at index 1 (index 0 is intercept)
                period_coefs[col].append(res.params[i + 1])
        except Exception as exc:
            logger.debug("FM period %s failed: %s", period, exc)
            continue

        valid_periods.append(period)

    records: list[dict] = []
    for col in x_columns:
        ts = pd.Series(period_coefs[col])
        mean_c = ts.mean()
        std_c = ts.std()
        n_periods = len(ts)
        tstat, pval = _newey_west_tstat(ts, lags=nw_lags)
        records.append(
            {
                "variable": col,
                "mean_coef": round(mean_c, 6),
                "std_coef": round(std_c, 6),
                "t_stat": round(tstat, 4) if not pd.isna(tstat) else np.nan,
                "p_value": round(pval, 4) if not pd.isna(pval) else np.nan,
                "n_periods": n_periods,
            }
        )

    result_df = pd.DataFrame(records)
    logger.info(
        "Fama-MacBeth [%s]: %d valid periods, %d regressors",
        y_column, len(valid_periods), len(x_columns),
    )
    return result_df
