import numpy as np
import pandas as pd


def compute_stateful_signal_with_decay(
    series: pd.Series,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
) -> pd.Series:
    """
    Compute a stateful signal that locks in an overbought/oversold state when the
    rolling z-score exceeds the trigger threshold, then decays the signal over time.

    Vectorized implementation using NumPy for efficiency.

    Args:
        series (pd.Series): Price or return series.
        params (dict): Contains "window_daily" and "z_threshold_daily".
        target_decay (float): Fraction of the original signal remaining after optimal_window days.
        reset_factor (float): Factor to determine reset_threshold from trigger_threshold.

    Returns:
        pd.Series: Stateful signal time series.
    """
    window = int(params.get("window_daily", 20))
    trigger_threshold = params.get("z_threshold_daily", 1.5)
    optimal_window = window  # Assume optimal window = rolling window
    reset_threshold = trigger_threshold * reset_factor
    decay_rate = target_decay ** (1 / optimal_window)

    # Compute rolling z-scores
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = (
        series.rolling(window=window, min_periods=window).std().replace(0, np.nan)
    )
    z_scores = (series - rolling_mean) / rolling_std

    # Initialize state tracking arrays
    state = np.zeros(len(series))  # 0: neutral, -1: overbought, +1: oversold
    state_age = np.zeros(len(series))
    signal = np.zeros(len(series))

    # Identify trigger events (first time z-score crosses threshold)
    overbought = z_scores > trigger_threshold
    oversold = z_scores < -trigger_threshold

    # Iterate through the array vectorized
    for i in range(1, len(series)):
        if np.isnan(z_scores.iloc[i]):
            continue

        if state[i - 1] == 0:
            # Check if we trigger an overbought/oversold state
            if overbought.iloc[i]:
                state[i] = -1  # Overbought
                state_age[i] = 0
            elif oversold.iloc[i]:
                state[i] = 1  # Oversold
                state_age[i] = 0
        else:
            # Maintain state
            state[i] = state[i - 1]
            state_age[i] = state_age[i - 1] + 1

            # Reset if z-score falls below reset threshold
            if state[i] == -1 and z_scores.iloc[i] < reset_threshold:
                state[i] = 0
                state_age[i] = 0
            elif state[i] == 1 and z_scores.iloc[i] > -reset_threshold:
                state[i] = 0
                state_age[i] = 0

        # Compute decay multiplier and apply stateful signal
        decay_multiplier = decay_rate ** state_age[i] if state[i] != 0 else 0
        signal_magnitude = (
            abs(z_scores.iloc[i]) if abs(z_scores.iloc[i]) >= trigger_threshold else 0
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
    # Compute daily signal
    daily_signal = compute_stateful_signal_with_decay(
        ticker_series, params, target_decay=target_decay, reset_factor=reset_factor
    )

    # Compute weekly signal
    weekly_series = ticker_series.resample("W").last()
    weekly_signal = compute_stateful_signal_with_decay(
        weekly_series, params, target_decay=target_decay, reset_factor=reset_factor
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
