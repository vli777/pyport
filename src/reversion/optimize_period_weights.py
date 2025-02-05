from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from reversion.strategy_metrics import composite_score, simulate_strategy
from utils.logger import logger


def find_optimal_weights(
    daily_signals_df: pd.DataFrame,
    weekly_signals_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    reoptimize: bool = False,
    cache: dict = None,  # Global cache dict keyed by ticker.
    group_id: str = None,  # Identifier for the cluster group.
) -> Dict[str, float]:
    """
    Run Optuna to find the optimal weighting of daily and weekly signals.
    This function first checks whether the tickers in the group already have period
    weights stored in the global cache. If so, it returns those weights directly.

    Args:
        daily_signals_df (pd.DataFrame): Daily signals DataFrame.
        weekly_signals_df (pd.DataFrame): Weekly signals DataFrame.
        returns_df (pd.DataFrame): Log returns DataFrame (its columns should be the tickers in the group).
        n_trials (int, optional): Number of optimization trials. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        reoptimize (bool): If True, forces reoptimization.
        cache (dict, optional): Global cache dict keyed by ticker.
        group_id (str): Identifier for the cluster group.

    Returns:
        Dict[str, float]: Best weights for daily and weekly signals for the group.
    """
    # Extract tickers in the current group from returns_df columns.
    tickers_in_group = list(returns_df.columns)

    # Check if every ticker in this group already has "weight_daily" in its cache entry.
    if (
        not reoptimize
        and cache is not None
        and all(t in cache and "weight_daily" in cache[t] for t in tickers_in_group)
    ):
        # Use the weights from the first ticker (they should be identical within the group).
        group_params = cache[tickers_in_group[0]]
        return {
            "weight_daily": group_params.get("weight_daily", 0.7),
            "weight_weekly": group_params.get("weight_weekly", 0.3),
        }

    # Otherwise, perform optimization.
    study = optuna.create_study(
        study_name=f"reversion_weights_optimization_{group_id}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=max(5, n_trials // 10)),
    )

    study.optimize(
        lambda trial: reversion_weights_objective(
            trial, daily_signals_df, weekly_signals_df, returns_df
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    if study.best_trial is None:
        logger.error("No valid optimization results found.")
        best_weights = {"weight_daily": 0.7, "weight_weekly": 0.3}
    else:
        best_weights = study.best_trial.params

    # At this point, the caller is responsible for updating the global cache for all tickers in the group.
    return best_weights


def reversion_weights_objective(
    trial,
    daily_signals_df: pd.DataFrame,
    weekly_signals_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> float:
    """
    Optimize the weights of different time scale signals for mean reversion.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        daily_signals_df (pd.DataFrame): Daily signals DataFrame.
        weekly_signals_df (pd.DataFrame): Weekly signals DataFrame.
        returns_df (pd.DataFrame): Log returns DataFrame.

    Returns:
        float: composite score from returns and performance ratios for the optimized strategy.
    """
    weight_daily = trial.suggest_float("weight_daily", 0.0, 1.0, step=0.1)
    weight_weekly = 1.0 - weight_daily

    # Combine the precomputed dataframes using vectorized operations
    combined = weight_daily * daily_signals_df + weight_weekly * weekly_signals_df

    # Vectorize the mapping to discrete signals
    combined_signals = pd.DataFrame(
        np.sign(combined.values),
        index=combined.index,
        columns=combined.columns,
    )

    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    combined_signals = combined_signals[valid_stocks]

    aligned_returns = returns_df[valid_stocks].reindex(combined_signals.index)
    positions_df = combined_signals.shift(1).fillna(0)

    # Run the simulation and calculate a composite score.
    _, metrics = simulate_strategy(aligned_returns, positions_df)

    return composite_score(metrics)
