from typing import Dict, List, Tuple
import pandas as pd


def generate_reversion_recommendations(
    reversion_signals: Dict[str, Dict[str, Dict[str, int]]],
    optimal_weights: Dict[str, float],
    include_pct: float = 0.2,  # Top 20% buy
    exclude_pct: float = 0.2,  # Bottom 20% sell
) -> Dict[str, List[str]]:
    """
    Generate final tickers to include/exclude based on dynamically determined thresholds.

    Args:
        reversion_signals (Dict[str, Dict[str, Dict[str, int]]]):
            Dictionary of buy/sell signals per timeframe.
        optimal_weights (Dict[str, float]): Optimal weights for each timeframe.
        include_pct (float, optional): Top X% of signals to include. Defaults to 20%.
        exclude_pct (float, optional): Bottom X% of signals to exclude. Defaults to 20%.

    Returns:
        Dict[str, List[str]]: Dictionary with "include" and "exclude" tickers.
    """
    # Convert signals to DataFrames
    daily_signals = pd.DataFrame.from_dict(reversion_signals["daily"])
    weekly_signals = pd.DataFrame.from_dict(reversion_signals["weekly"])

    # Compute weighted signal strength
    weight_daily = optimal_weights["weight_daily"]
    weight_weekly = 1.0 - weight_daily
    final_signals = (
        weight_daily * daily_signals + weight_weekly * weekly_signals
    ).mean(axis=1)

    # Compute dynamic thresholds based on percentiles
    include_threshold = final_signals.quantile(1 - include_pct)  # Top X% for buys
    exclude_threshold = final_signals.quantile(exclude_pct)  # Bottom X% for sells

    # Select tickers dynamically
    include = set(final_signals[final_signals >= include_threshold].index.tolist())
    exclude = set(final_signals[final_signals <= exclude_threshold].index.tolist())

    # Remove conflicts: If a ticker is in both, remove it from both
    conflicting_tickers = include.intersection(exclude)
    if conflicting_tickers:
        print(f"{len(conflicting_tickers)} conflicting tickers removed.")
    include -= conflicting_tickers
    exclude -= conflicting_tickers

    return {"include": list(include), "exclude": list(exclude)}
