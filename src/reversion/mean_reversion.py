from typing import Any, Dict, Tuple
import pandas as pd
from config import Config

from reversion.cluster_mean_reversion import cluster_mean_reversion
from reversion.reversion_utils import (
    adjust_allocation_with_mean_reversion,
    calculate_continuous_composite_signal,
    group_ticker_params_by_cluster,
    propagate_signals_by_similarity,
)
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def apply_mean_reversion(
    baseline_allocation: pd.Series,
    returns_df: pd.DataFrame,
    config: Config,
    cache_dir: str = "optuna_cache",
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Generate continuous mean reversion signals on clusters of stocks and overlay the adjustment
    onto the baseline allocation using a continuous adjustment factor.

    This version loads a global cache (a dictionary of ticker-level parameters) and passes it
    to cluster_mean_reversion. That function updates the cache for any tickers that need optimization.
    Later, the composite signals are computed using these cached parameters.

    Args:
        baseline_allocation (pd.Series): Original weight allocation after optimization.
        returns_df (pd.DataFrame): Returns for all selected stocks.
        config (Config): Configuration object.
        cache_dir (str): Directory for cache files.

    Returns:
        pd.Series: Final adjusted allocation.
    """
    # Load (or initialize) the global cache.
    global_cache_file = f"{cache_dir}/reversion_params_global.pkl"
    global_cache = load_parameters_from_pickle(global_cache_file)
    if not isinstance(global_cache, dict):
        global_cache = {}

    # Update the global cache via clustering & optimization.
    previous_cache = global_cache.copy()
    group_reversion_signals = cluster_mean_reversion(
        returns_df,
        n_trials=50,
        n_jobs=-1,
        global_cache=global_cache,
    )
    print("Reversion Signals Generated.")

    # Save the updated global cache only if changes occurred.
    if global_cache != previous_cache:
        save_parameters_to_pickle(global_cache, global_cache_file)

    ticker_params = global_cache
    print(f"Loaded Ticker Parameters for {len(ticker_params)} tickers.")

    # Use the global cache (ticker_params) to compute composite signals.
    composite_signals = calculate_continuous_composite_signal(
        group_signals=group_reversion_signals, ticker_params=ticker_params
    )

    # Propagate signals based on pairwise similarity:
    group_mapping = group_ticker_params_by_cluster(ticker_params)
    updated_composite_signals = propagate_signals_by_similarity(
        composite_signals=composite_signals,
        group_mapping=group_mapping,
        baseline_allocation=baseline_allocation,
        returns_df=returns_df,
        signal_dampening=0.5,
        lw_threshold=50,
    )

    # Adjust the baseline allocation using the updated composite signals.
    final_allocation = adjust_allocation_with_mean_reversion(
        baseline_allocation=baseline_allocation,
        composite_signals=updated_composite_signals,
        alpha=config.mean_reversion_strength,
        allow_short=config.allow_short,
    )

    return final_allocation, composite_signals
