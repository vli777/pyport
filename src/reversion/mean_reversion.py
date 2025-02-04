import pandas as pd
from config import Config
from reversion.cluster_mean_reversion import cluster_mean_reversion
from reversion.optimize_period_weights import optimize_group_weights
from reversion.reversion_utils import (
    adjust_allocation_with_mean_reversion,
    calculate_continuous_composite_signal,
)
from reversion.plot_reversion_clusters import plot_reversion_clusters


def apply_mean_reversion(
    baseline_allocation: pd.Series, returns_df: pd.DataFrame, config: Config
) -> pd.Series:
    """
    Generate continuous mean reversion signals on clusters of stocks and overlay the adjustment
    onto the baseline allocation using a continuous adjustment factor.

    The continuous signal (e.g. a z-score) is used to adjust the baseline weight via:
        new_weight = baseline_weight * (1 + alpha * continuous_signal)
    followed by renormalization.

    Args:
        baseline_allocation (pd.Series): Original weight allocation after optimization.
        returns_df (pd.DataFrame): Returns for all selected stocks.
        config (Config): Configuration object.

    Returns:
        pd.Series: Final adjusted allocation.
    """
    # Step 1. Cluster the stocks and compute continuous reversion signals per cluster.
    group_reversion_signals = cluster_mean_reversion(
        returns_df,
        n_trials=50,
        n_jobs=-1,
        cache_dir="optuna_cache",
        reoptimize=False,
    )
    print("Reversion Signals Generated.")

    # Step 2. Optimize the weighting parameters to combine daily/weekly signals for each cluster.
    optimal_period_weights = optimize_group_weights(
        group_reversion_signals,
        returns_df,
        n_trials=50,
        n_jobs=-1,
        reoptimize=False,
    )
    print(f"Optimal Period Weights: {optimal_period_weights}")

    # Optional: if plotting is enabled, visualize the clusters and parameters.
    if config.plot_reversion:
        plot_reversion_clusters(
            returns_df=returns_df,
            group_reversion_signals=group_reversion_signals,
            optimal_period_weights=optimal_period_weights,
            title="Mean Reversion Groups & Parameters",
        )

    # Step 3. Compute the continuous composite signal from the clusters.
    composite_signals = calculate_continuous_composite_signal(
        group_signals=group_reversion_signals, group_weights=optimal_period_weights
    )
    print(f"Composite Signals: {composite_signals}")

    # Step 4. Overlay the continuous mean reversion adjustment onto the baseline allocation.
    # Here, alpha controls the influence of the signal.
    final_allocation = adjust_allocation_with_mean_reversion(
        baseline_allocation=baseline_allocation,
        composite_signals=composite_signals,
        alpha=config.mean_reversion_strength,  # e.g. 0.2
        allow_short=config.allow_short,
    )

    return final_allocation
