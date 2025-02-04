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
    cache: dict = None,  # Optional global cache for weights.
) -> Dict[str, float]:
    """
    Run Optuna to find the optimal weighting of daily and weekly signals.
    If a cache dict is provided and contains weights, they are used.

    Args:
        daily_signals_df (pd.DataFrame): Daily signals DataFrame.
        weekly_signals_df (pd.DataFrame): Weekly signals DataFrame.
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        reoptimize (bool): Override to reoptimize.
        cache (dict, optional): A global cache dict; if provided, its "weights" key is used.

    Returns:
        Dict[str, float]: Best weights for daily and weekly signals.
    """
    if not reoptimize and cache is not None and "weights" in cache:
        return cache["weights"]

    study = optuna.create_study(
        study_name="reversion_weights_optimization",
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
        best_weights = {"weight_daily": 0.5, "weight_weekly": 0.5}
    else:
        best_weights = study.best_trial.params

    if cache is not None:
        cache["weights"] = best_weights

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
