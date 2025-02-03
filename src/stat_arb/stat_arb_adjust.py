from typing import Dict
import pandas as pd


def calculate_composite_signal(
    reversion_signals: Dict[str, Dict[str, Dict[str, int]]],
    weight_daily: float = 0.5,
    weight_weekly: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a composite stat arb signal from reversion signals.
    For each ticker, use the latest available signal from the daily and weekly signals.
    If a ticker is missing a signal in one timeframe, assume it is 0.

    Args:
        reversion_signals (dict): Dict with keys "daily" and "weekly". Each maps ticker -> {date: signal}.
        weight_daily (float): Weight for the daily signal.
        weight_weekly (float): Weight for the weekly signal.

    Returns:
        dict: Mapping from ticker to composite signal (a float).
    """
    composite = {}
    # Get the union of tickers present in either timeframe.
    tickers = set(reversion_signals.get("daily", {}).keys()) | set(
        reversion_signals.get("weekly", {}).keys()
    )

    for ticker in tickers:
        # For daily signals: get the latest available value.
        daily_signals = reversion_signals.get("daily", {}).get(ticker, {})
        daily_val = 0
        if daily_signals:
            latest_daily = max(
                daily_signals.keys()
            )  # Assumes date-like keys that can be compared.
            daily_val = daily_signals[latest_daily]

        # For weekly signals: similarly, get the latest value.
        weekly_signals = reversion_signals.get("weekly", {}).get(ticker, {})
        weekly_val = 0
        if weekly_signals:
            latest_weekly = max(weekly_signals.keys())
            weekly_val = weekly_signals[latest_weekly]

        composite[ticker] = weight_daily * daily_val + weight_weekly * weekly_val

    return composite


def adjust_allocation_with_stat_arb(
    baseline_allocation: pd.Series,
    composite_signals: Dict[str, float],
    alpha: float = 0.2,
    allow_short: bool = False,
) -> pd.Series:
    """
    Adjust the baseline allocation weights using the composite stat arb signal.
    The adjustment is multiplicative:
        new_weight = baseline_weight * (1 + alpha * composite_signal)
    Optionally, if shorts are not allowed, negative adjusted weights are clipped to zero.
    Finally, the weights are renormalized to sum to one.

    Args:
        baseline_allocation (pd.Series): Series with index = ticker and values = baseline weights.
        composite_signals (dict): Mapping from ticker to composite signal (float).
        alpha (float): Sensitivity factor controlling how strongly the signal adjusts the weight.
        allow_short (bool): Whether negative weights (shorts) are allowed. If False, negative weights are set to 0.

    Returns:
        pd.Series: The adjusted and normalized allocation weights.
    """
    adjusted = baseline_allocation.copy()
    for ticker in adjusted.index:
        signal = composite_signals.get(ticker, 0)
        # Multiply the baseline weight by a factor that increases with a positive signal.
        adjusted[ticker] = adjusted[ticker] * (1 + alpha * signal)

    # If shorts are not allowed, clip negative weights to zero.
    if not allow_short:
        adjusted = adjusted.clip(lower=0)

    # Renormalize so that the weights sum to 1.
    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total
    return adjusted
