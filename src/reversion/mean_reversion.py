import pandas as pd
from config import Config
from reversion.cluster_mean_reversion import cluster_mean_reversion
from reversion.optimize_period_weights import optimize_group_weights
from reversion.reversion_utils import (
    adjust_allocation_with_mean_reversion,
    calculate_composite_signal,
)
from reversion.plot_reversion_clusters import plot_reversion_clusters


def apply_mean_reversion(
    baseline_allocation: pd.Series, returns_df: pd.DataFrame, config: Config
) -> pd.Series:
    """
    Apply z-score based allocation weight adjustments.

    Args:
        baseline_allocation (pd.Series): Original weight allocation after optimization.
        returns_df (pd.DataFrame): The returns dataframe of all selected stocks.
        config (Config): yaml config loader object.

    Returns:
        pd.Series: Final adjusted allocation.
    """
    # Step 1. Generate reversion signals for each similar asset cluster.
    group_reversion_signals = cluster_mean_reversion(
        returns_df,
        n_trials=50,
        n_jobs=-1,
        cache_dir="optuna_cache",
        reoptimize=False,
    )
    print("Reversion Signals Generated.")

    # Step 2. Optimize weights for the reversion signals (daily/weekly weighting) per group.
    optimal_period_weights = optimize_group_weights(
        group_reversion_signals,
        returns_df,
        n_trials=50,
        n_jobs=-1,
        reoptimize=False,
    )
    print(f"Optimal Period Weights: {optimal_period_weights}")

    # If plotting is enabled, produce the cluster visualization.
    if config.plot_reversion:
        plot_reversion_clusters(
            returns_df=returns_df,
            group_reversion_signals=group_reversion_signals,
            optimal_period_weights=optimal_period_weights,
            title="Mean Reversion Groups & Parameters",
        )

    # Step 3. Compute the composite stat arb signal from the reversion signals.
    composite_signals = calculate_composite_signal(
        group_signals=group_reversion_signals, group_weights=optimal_period_weights
    )
    print(f"Composite Signals: {composite_signals}")

    # Step 4. Adjust allocation using the composite signals.
    final_allocation = adjust_allocation_with_mean_reversion(
        baseline_allocation=baseline_allocation,
        composite_signals=composite_signals,
        alpha=0.2,
        allow_short=config.allow_short,
    )

    return final_allocation
