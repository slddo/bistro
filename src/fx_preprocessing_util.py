"""
FX (Foreign Exchange) rate preprocessing utilities for BISTRO.

Adapts the macroeconomic preprocessing pipeline for exchange rate forecasting.
The main difference from the CPI pipeline: no year-over-year transformation.
FX rates are used as raw levels or optionally converted to log returns.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from preprocessing_util import (
    DailyInferencePrep,
    _standardize_period_index,
    _period_to_period_end_timestamp,
    pad_future_markers,
    forward_fill_to_daily,
    detect_and_impute_gaps,
    prepare_yoy_monthly_for_daily_inference,
)


def compute_log_returns(series: pd.Series, *, scale: float = 100.0) -> pd.Series:
    """Convert FX rate levels to log returns: scale * ln(p_t / p_{t-1})."""
    return (np.log(series / series.shift(1)) * scale).rename(series.name)


def compute_rate_differential(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    name: str = "rate_differential",
) -> pd.Series:
    """
    Compute the interest rate differential on the common date range.

    Typically: US policy rate minus a foreign policy rate.
    Used as a covariate via uncovered interest rate parity logic.
    """
    aligned_a, aligned_b = series_a.align(series_b, join="inner")
    return (aligned_a - aligned_b).rename(name)


def prepare_fx_monthly_for_inference(
    df_fx: pd.DataFrame,
    *,
    target_col: str,
    freq: str = "M",
    use_log_returns: bool = False,
    forecast_start_date: str,
    pdt_patches: int,
    ctx_patches: int,
    steps_per_period: int = 32,
    rolling_windows: int = 4,
    window_distance_patches: int = 2,
    tolerance_days: int = 10,
) -> DailyInferencePrep:
    """
    Prepare a monthly FX rate series for BISTRO daily inference (univariate).

    Unlike the CPI macro pipeline, no YoY transformation is applied.
    Set use_log_returns=True to forecast log-return changes instead of levels.

    Parameters
    ----------
    df_fx : DataFrame with period or datetime index and FX rate column
    target_col : column name of the FX rate to forecast
    freq : pandas frequency string ('M' for monthly, 'Q' for quarterly)
    use_log_returns : convert levels to log returns (× 100) before forecasting
    forecast_start_date : start date of the first forecast window ('YYYY-MM-DD')
    pdt_patches : forecast horizon in periods
    ctx_patches : model context length in periods
    steps_per_period : patch size in days (32 maps one calendar month to one patch)
    rolling_windows : number of rolling backtesting windows
    window_distance_patches : spacing in periods between consecutive windows
    tolerance_days : maximum allowed timestamp snap for alignment

    Returns
    -------
    DailyInferencePrep — same structure as the macro pipeline for compatibility
    """
    df = df_fx[[target_col]].copy()

    if use_log_returns:
        df[target_col] = compute_log_returns(df[target_col])
        df = df.dropna()

    return prepare_yoy_monthly_for_daily_inference(
        df,
        target_col=target_col,
        freq=freq,
        forecast_start_date=forecast_start_date,
        pdt_patches=pdt_patches,
        ctx_patches=ctx_patches,
        steps_per_period=steps_per_period,
        rolling_windows=rolling_windows,
        window_distance_patches=window_distance_patches,
        tolerance_days=tolerance_days,
    )


def prepare_live_forecast(
    df_monthly: pd.DataFrame,
    *,
    target_col: str,
    freq: str = "M",
    use_log_returns: bool = False,
    ctx_patches: int = 120,
    steps_per_period: int = 32,
    tolerance_days: int = 10,
) -> Tuple[pd.DataFrame, "pd.Period", int]:
    """
    Prepare data for a single live forecast from the last available data point.

    Because there is no future actual data, the standard rolling-window approach
    cannot be used.  This function pads one period of marker values after the
    last observation so that PandasDataset + split() + generate_instances()
    can create exactly one inference window that forecasts the next period.

    Parameters
    ----------
    df_monthly : DataFrame with period/datetime index and target FX rate column
    target_col : column name of the FX rate to forecast
    freq : pandas frequency string ('M' for monthly)
    use_log_returns : if True, convert levels to log returns before forecasting
    ctx_patches : number of periods of history for the model context
    steps_per_period : patch size in days (32 ≈ one calendar month)
    tolerance_days : max allowed timestamp snap for alignment

    Returns
    -------
    (daily_df, cutoff_daily, ctx_steps)
      daily_df     : padded daily DataFrame ready for PandasDataset
      cutoff_daily : pd.Period (daily freq) to pass to split()
      ctx_steps    : integer context steps for generate_instances()
    """
    df = df_monthly[[target_col]].copy()

    if use_log_returns:
        df[target_col] = compute_log_returns(df[target_col])
        df = df.dropna()

    # Standardise to PeriodIndex then convert to DatetimeIndex
    df.index = _standardize_period_index(df.index, freq=freq)
    df_dt = df.copy()
    if freq == "M":
        df_dt.index = df_dt.index.to_timestamp(freq="M")
    elif freq == "Q":
        df_dt.index = df_dt.index.to_timestamp(freq="Q")
    else:
        df_dt.index = df_dt.index.to_timestamp()

    df_dt = detect_and_impute_gaps(df_dt.dropna(), freq=freq, tolerance_days=tolerance_days)

    # Pad one future period with marker values so the split point falls inside
    padded = pad_future_markers(df_dt, target_col=target_col, n_pad_periods=1, freq=freq)
    daily_df = forward_fill_to_daily(padded, patch_size_days=steps_per_period)

    # Cutoff = last day of the last actual month
    last_actual_ts = df_dt.index[-1]
    cutoff_daily = pd.Period(last_actual_ts.strftime("%Y-%m-%d"))
    ctx_steps = ctx_patches * steps_per_period

    return daily_df, cutoff_daily, ctx_steps
