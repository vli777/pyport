from typing import Dict, List, Tuple
import pandas as pd

from utils.logger import logger


def generate_reversion_recommendations(
    reversion_signals: Dict[str, Dict[str, Dict[str, int]]],
    optimal_weights: Dict[str, float],
    include_pct: float = 0.2,  # Top 20% buy
    exclude_pct: float = 0.2,  # Bottom 20% sell
) -> Dict[str, List[str]]:
    """
    Generate final recommendations by selecting tickers for inclusion or exclusion.
    Uses the weighted average of daily and weekly signals and picks the top and bottom percentiles.

    Args:
        reversion_signals (Dict[str, Dict[str, Dict[str, int]]]):
            Dictionary of buy/sell signals per timeframe.
        optimal_weights (Dict[str, float]): Optimal weights for each timeframe.
        include_pct (float, optional): Top X% of signals to include. Defaults to 20%.
        exclude_pct (float, optional): Bottom X% of signals to exclude. Defaults to 20%.

    Returns:
        Dict[str, List[str]]: Dictionary with "include" and "exclude" tickers.
    """
    daily_signals = pd.DataFrame.from_dict(reversion_signals["daily"])
    weekly_signals = pd.DataFrame.from_dict(reversion_signals["weekly"])

    weight_daily = optimal_weights.get("weight_daily", 0.5)
    weight_weekly = 1.0 - weight_daily
    final_signals = (
        weight_daily * daily_signals + weight_weekly * weekly_signals
    ).mean(axis=1)

    include_threshold = final_signals.quantile(1 - include_pct)
    exclude_threshold = final_signals.quantile(exclude_pct)

    include = set(final_signals[final_signals >= include_threshold].index.tolist())
    exclude = set(final_signals[final_signals <= exclude_threshold].index.tolist())

    # Remove any tickers that appear in both lists
    conflicting = include.intersection(exclude)
    if conflicting:
        logger.info(f"{len(conflicting)} conflicting tickers removed.")
    include -= conflicting
    exclude -= conflicting

    return {"include": list(include), "exclude": list(exclude)}
