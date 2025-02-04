import pandas as pd
from config import Config
from reversion.cluster_mean_reversion import cluster_mean_reversion
from reversion.reversion_utils import (
    adjust_allocation_with_mean_reversion,
    calculate_continuous_composite_signal,
)
from reversion.reversion_plots import plot_reversion_scatter
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def apply_mean_reversion(
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
    group_reversion_signals = cluster_mean_reversion(
        returns_df,
        n_trials=50,
        n_jobs=-1,
        reoptimize=False,
        global_cache=global_cache,
    )
    print("Reversion Signals Generated.")

    # Save the updated global cache.
    save_parameters_to_pickle(global_cache, global_cache_file)
    ticker_params = global_cache
    print(f"Loaded Ticker Parameters for {len(ticker_params)} tickers.")

    if config.plot_reversion:
        plot_reversion_scatter(
            ticker_params=ticker_params, title="Mean Reversion Parameters"
        )

    # Use the global cache (ticker_params) to compute composite signals.
    composite_signals = calculate_continuous_composite_signal(
        group_signals=group_reversion_signals, ticker_params=ticker_params
    )
    print(f"Composite Signals: {composite_signals}")

    # Adjust the baseline allocation.
    final_allocation = adjust_allocation_with_mean_reversion(
        baseline_allocation=baseline_allocation,
        composite_signals=composite_signals,
        alpha=config.mean_reversion_strength,
        allow_short=config.allow_short,
    )

    return final_allocation
