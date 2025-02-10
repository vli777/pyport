import numpy as np
import pandas as pd


def compute_stateful_signal_with_decay(
    series: pd.Series,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
) -> pd.Series:
    """
    Compute a stateful signal with separate thresholds for overbought (short) and oversold (long)
    conditions. For instance, the state is triggered to -1 (overbought/short) when the z-score
    exceeds z_threshold_positive, and to +1 (oversold/long) when the z-score is below -z_threshold_negative.
    The signal then decays over time.

    Args:
        series (pd.Series): Price or return series.
        params (dict): Should contain:
            - "window": rolling window size,
            - "z_threshold_positive": threshold for triggering an overbought state,
            - "z_threshold_negative": threshold for triggering an oversold state.
        target_decay (float): Fraction of the original signal remaining after optimal_window days.
        reset_factor (float): Factor to derive the reset threshold from the trigger threshold.

    Returns:
        pd.Series: The stateful signal time series.
    """
    window = int(params.get("window", 20))
    trigger_threshold_pos = params.get("z_threshold_positive", 1.5)
    trigger_threshold_neg = params.get("z_threshold_negative", 1.5)

    # Define reset thresholds for each side.
    reset_threshold_pos = trigger_threshold_pos * reset_factor
    reset_threshold_neg = trigger_threshold_neg * reset_factor

    optimal_window = window  # For simplicity, assume the same window applies for decay
    decay_rate = target_decay ** (1 / optimal_window)

    # Compute rolling z-scores.
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = (
        series.rolling(window=window, min_periods=window).std().replace(0, np.nan)
    )
    z_scores = (series - rolling_mean) / rolling_std

    # Initialize arrays.
    state = np.zeros(
        len(series)
    )  # 0: neutral, -1: overbought (short), +1: oversold (long)
    state_age = np.zeros(len(series))
    signal = np.zeros(len(series))

    # Identify trigger conditions.
    # For overbought, we need z > trigger_threshold_pos.
    # For oversold, we need z < -trigger_threshold_neg.
    for i in range(1, len(series)):
        if np.isnan(z_scores.iloc[i]):
            continue

        if state[i - 1] == 0:
            if z_scores.iloc[i] > trigger_threshold_pos:
                state[i] = -1  # Trigger overbought/short state.
                state_age[i] = 0
            elif z_scores.iloc[i] < -trigger_threshold_neg:
                state[i] = 1  # Trigger oversold/long state.
                state_age[i] = 0
        else:
            # Continue previous state.
            state[i] = state[i - 1]
            state_age[i] = state_age[i - 1] + 1

            # Reset conditions.
            if state[i] == -1 and z_scores.iloc[i] < reset_threshold_pos:
                state[i] = 0
                state_age[i] = 0
            elif state[i] == 1 and z_scores.iloc[i] > -reset_threshold_neg:
                state[i] = 0
                state_age[i] = 0

        # Compute decayed signal.
        decay_multiplier = decay_rate ** state_age[i] if state[i] != 0 else 0
        # Use the appropriate threshold for checking if the signal magnitude qualifies.
        if state[i] == -1:
            thresh = trigger_threshold_pos
        elif state[i] == 1:
            thresh = trigger_threshold_neg
        else:
            thresh = 0

        signal_magnitude = (
            abs(z_scores.iloc[i]) if abs(z_scores.iloc[i]) >= thresh else 0
        )
        signal[i] = state[i] * signal_magnitude * decay_multiplier

    return pd.Series(signal, index=series.index)


def compute_ticker_stateful_signals(
    ticker_series: pd.Series,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
) -> dict:
    """
    Compute stateful signals for a ticker on both daily and weekly data.

    Returns:
        dict: {
            "daily": pd.Series,
            "weekly": pd.Series
        }
    """
    # Build daily parameters dictionary.
    daily_params = {
        "window": int(params.get("window_daily", 20)),
        "z_threshold_positive": params.get("z_threshold_daily_positive", 1.5),
        "z_threshold_negative": params.get("z_threshold_daily_negative", 1.5),
    }
    daily_signal = compute_stateful_signal_with_decay(
        ticker_series,
        daily_params,
        target_decay=target_decay,
        reset_factor=reset_factor,
    )

    # Build weekly parameters dictionary.
    weekly_series = ticker_series.resample("W").last()
    weekly_params = {
        "window": int(params.get("window_weekly", 5)),
        "z_threshold_positive": params.get("z_threshold_weekly_positive", 1.5),
        "z_threshold_negative": params.get("z_threshold_weekly_negative", 1.5),
    }
    weekly_signal = compute_stateful_signal_with_decay(
        weekly_series,
        weekly_params,
        target_decay=target_decay,
        reset_factor=reset_factor,
    )

    return {"daily": daily_signal, "weekly": weekly_signal}


def compute_group_stateful_signals(
    group_returns: pd.DataFrame,
    tickers: list,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
) -> dict:
    """
    Given a group of tickers and their returns, compute stateful signals for each ticker.

    Returns:
        dict: Mapping from ticker to its signals, e.g.
            {
                "AAPL": {"daily": {date: signal, ...}, "weekly": {date: signal, ...}},
                "MSFT": {...},
                ...
            }
    """
    signals = {}
    for ticker in tickers:
        series = group_returns[ticker].dropna()
        if series.empty:
            signals[ticker] = {"daily": {}, "weekly": {}}
        else:
            signals[ticker] = compute_ticker_stateful_signals(
                series, params, target_decay=target_decay, reset_factor=reset_factor
            )
    return signals
