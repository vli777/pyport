import numpy as np
import pandas as pd


def compute_stateful_signal_with_decay(
    series, params, target_decay=0.5, reset_factor=0.5
):
    """
    Compute a stateful signal that locks in an overbought/oversold state when the
    rolling z-score exceeds the trigger threshold, then decays the signal over time.

    The function uses the group parameters:
      - "window_daily": rolling window size for computing the z-score.
      - "z_threshold_daily": the threshold determined (by optuna) that triggers a state.

    From these, we derive:
      - trigger_threshold = z_threshold_daily
      - reset_threshold = trigger_threshold * reset_factor
      - optimal_window = window_daily (assumed for decay timing)
      - decay_rate is chosen so that after optimal_window days the signal decays to target_decay of its initial value.

    Args:
        series (pd.Series): Price or return series.
        params (dict): Should contain "window_daily" and "z_threshold_daily".
        target_decay (float): Fraction of the original signal remaining after optimal_window days.
                              For example, 0.5 means the signal should be about 50% after optimal_window days.
        reset_factor (float): Factor to determine reset_threshold from trigger_threshold.
                              For example, 0.5 means reset_threshold = trigger_threshold * 0.5.

    Returns:
        dict: Mapping of date -> stateful signal.
    """
    window = int(params.get("window_daily", 20))
    trigger_threshold = params.get("z_threshold_daily", 1.5)
    optimal_window = window  # We assume the optimal reversion window is the same as the rolling window.
    reset_threshold = trigger_threshold * reset_factor

    state = 0  # 0: neutral, -1: overbought, +1: oversold.
    state_age = 0  # Number of days since the signal was locked in.
    stateful_signal = {}

    # Compute the daily z-scores.
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = (
        series.rolling(window=window, min_periods=window).std().replace(0, np.nan)
    )
    z_scores = (series - rolling_mean) / rolling_std

    # Compute the daily decay rate such that after 'optimal_window' days the signal decays to target_decay of its initial value.
    decay_rate = target_decay ** (1 / optimal_window)

    for date, z in z_scores.items():
        if pd.isna(z):
            stateful_signal[date] = 0
            continue

        if state == 0:
            # Check for trigger: if z exceeds the threshold (positive or negative).
            if z > trigger_threshold:
                state = -1  # Overbought state; represent as -1.
                state_age = 0
            elif z < -trigger_threshold:
                state = 1  # Oversold state; represent as +1.
                state_age = 0
        else:
            state_age += 1
            # Reset the state if the z-score has decayed to below the reset threshold.
            if state == -1 and z < reset_threshold:
                state = 0
                state_age = 0
            elif state == 1 and z > -reset_threshold:
                state = 0
                state_age = 0

        # Apply exponential decay to the signal.
        decay_multiplier = decay_rate**state_age if state != 0 else 0
        # Use the absolute z-score as the signal magnitude (only if it meets the trigger requirement).
        signal_magnitude = abs(z) if abs(z) >= trigger_threshold else 0
        stateful_signal[date] = state * signal_magnitude * decay_multiplier

    return stateful_signal


def compute_group_stateful_signals(
    group_returns: pd.DataFrame,
    tickers: list,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
) -> dict:
    """
    Given a group of tickers and their returns along with optimized parameters,
    compute the stateful daily signals (with decay) for each ticker.

    The parameters (in params) should include at least:
      - "window_daily": rolling window size for computing the z-score.
      - "z_threshold_daily": the trigger threshold from optimization.

    Optionally, you can also pass:
      - "target_decay": desired fraction remaining after the window (default 0.5).
      - "reset_factor": factor to derive reset_threshold (default 0.5).

    Returns a dictionary with keys:
       - "tickers": the list of tickers
       - "daily": dict mapping ticker -> {date: stateful signal}
    """
    daily_signals = {}

    for ticker in tickers:
        series = group_returns[ticker].dropna()
        if series.empty:
            daily_signals[ticker] = {}
        else:
            daily_signals[ticker] = compute_stateful_signal_with_decay(
                series, params, target_decay=target_decay, reset_factor=reset_factor
            )

    return {
        "tickers": tickers,
        "daily": daily_signals,
    }
