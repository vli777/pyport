from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from reversion.strategy_metrics import composite_score, simulate_strategy
from utils.z_scores import calculate_robust_zscores
from utils.logger import logger


def optimize_robust_mean_reversion(
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    reoptimize: bool = False,
    cache: dict = None,  # Global cache dict passed in
    group_id: str = None,  # Extra parameter for the cluster identifier
) -> Tuple[Dict[str, float], optuna.study.Study]:
    """
    Optimize the rolling window and z_threshold using Optuna.
    If a cache dict is provided and it contains cached parameters for this group_id,
    those will be used.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        reoptimize (bool): Override to reoptimize.
        cache (dict, optional): A global cache dict.
        group_id (str): Identifier for the cluster group.

    Returns:
        Tuple[Dict[str, float], optuna.study.Study]: The best parameters and the study.
    """
    # Create a unique cache key using the group_id.
    cache_key = f"robust_params_{group_id}" if group_id is not None else "robust_params"

    if not reoptimize and cache is not None and cache_key in cache:
        logger.info(f"Using cached robust parameters for group {group_id}.")
        return cache[cache_key], None

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: robust_mean_reversion_objective(trial, returns_df),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    best_params = (
        study.best_trial.params
        if study.best_trial
        else {"window": 20, "z_threshold": 1.5}
    )

    if cache is not None:
        cache[cache_key] = best_params

    return best_params, study


def robust_mean_reversion_objective(trial, returns_df: pd.DataFrame) -> float:
    """
    Objective function for optimizing the robust mean reversion parameters.
    The trial suggests a rolling window size and a z_threshold.
    The resulting signals are used (with a one-day shift to avoid lookahead)
    to simulate a simple strategy; the cumulative return and sharpe ratio is maximized.
    """
    window = trial.suggest_int("window", 10, 30, step=5)
    z_threshold = trial.suggest_float("z_threshold", 1.0, 3.0, step=0.1)

    robust_z = calculate_robust_zscores(returns_df, window)
    # Generate signals on all tickers at once; result is a DataFrame of {date x ticker}
    signals = np.where(
        robust_z.values < -z_threshold,
        1,
        np.where(robust_z.values > z_threshold, -1, 0),
    )
    signals_df = pd.DataFrame(signals, index=robust_z.index, columns=robust_z.columns)

    positions_df = signals_df.shift(1).fillna(0)
    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    positions_df = positions_df[valid_stocks]
    aligned_returns = returns_df[valid_stocks].reindex(positions_df.index)

    _, metrics = simulate_strategy(aligned_returns, positions_df)

    return composite_score(metrics)
