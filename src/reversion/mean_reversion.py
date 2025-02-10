from datetime import datetime
from typing import Any, Dict, Tuple
import pandas as pd
from config import Config

from reversion.cluster_mean_reversion import cluster_mean_reversion
from reversion.reversion_utils import (
    adjust_allocation_with_mean_reversion,
    calculate_continuous_composite_signal,
    group_ticker_params_by_cluster,
    is_cache_stale,
    propagate_signals_by_similarity,
)
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def apply_mean_reversion(
    asset_cluster_map: Dict[str, int],
    baseline_allocation: pd.Series,
    returns_df: pd.DataFrame,
    config: Config,
    cache_dir: str = "optuna_cache",
) -> pd.Series:
    """
    Generate continuous mean reversion signals on clusters of stocks and overlay the adjustment
    onto the baseline allocation using a continuous adjustment factor.

    This version loads a global cache (a dictionary of ticker-level parameters) and passes it
    to cluster_mean_reversion. That function updates the cache for any tickers that need optimization.
    Later, the composite signals are computed using these cached parameters.

    Args:
        asset_cluster_map (dict): Map of assets to correlated clusters
        baseline_allocation (pd.Series): Original weight allocation after optimization.
        returns_df (pd.DataFrame): Returns for all selected stocks.
        config (Config): Configuration object.
        cache_dir (str): Directory for cache files.

    Returns:
        pd.Series: Final adjusted allocation.
    """
    # Load (or initialize) the global cache.
    reversion_cache_file = f"{cache_dir}/reversion_cache_global.pkl"
    reversion_cache = load_parameters_from_pickle(reversion_cache_file)
    if not isinstance(reversion_cache, dict):
        reversion_cache = {}

    # Ensure reversion_cache has a valid reference to params
    if "params" not in reversion_cache:
        reversion_cache["params"] = {}
    if "signals" not in reversion_cache:
        reversion_cache["signals"] = {}

    reversion_params = reversion_cache["params"]
    reversion_signals = reversion_cache["signals"]
    last_updated = reversion_cache.get("last_updated", "")

    # Ensure valid dictionary types
    if not isinstance(reversion_params, dict):
        reversion_params = reversion_cache["params"] = {}
    if not isinstance(reversion_signals, dict):
        reversion_signals = reversion_cache["signals"] = {}
    if not isinstance(last_updated, str) or not last_updated:
        last_updated = None  # Force a stale check

    # Identify missing tickers (those in `returns_df` but not in `reversion_signals`)
    missing_tickers = [
        ticker for ticker in returns_df.columns if ticker not in reversion_signals
    ]

    # Recalculate if cache is stale or signals are missing
    updated_reversion_signals = None
    if is_cache_stale(last_updated) or missing_tickers:
        updated_reversion_signals = cluster_mean_reversion(
            asset_cluster_map=asset_cluster_map,
            returns_df=returns_df[missing_tickers],
            n_trials=50,
            n_jobs=-1,
            global_cache=reversion_params,
        )
        # Update cache with newly computed signals
        reversion_signals.update(updated_reversion_signals)
        reversion_cache["last_updated"] = datetime.now().isoformat()

    # Save the updated global cache only if changes occurred.
    if updated_reversion_signals is not None:
        reversion_cache["signals"] = reversion_signals

    print("Reversion Signals Generated.")
    save_parameters_to_pickle(reversion_cache, reversion_cache_file)

    ticker_params = reversion_params
    print(f"Loaded Ticker Parameters for {len(ticker_params)} tickers.")

    # Use the global cache (ticker_params) to compute composite signals.
    composite_signals = calculate_continuous_composite_signal(
        signals=reversion_signals, ticker_params=ticker_params
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

    return final_allocation
