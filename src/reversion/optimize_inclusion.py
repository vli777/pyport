import os
from pathlib import Path
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import numpy as np
import optuna
import pandas as pd

from utils.caching_utils import (
    load_parameters_from_pickle,
    save_parameters_to_pickle,
)
from utils import logger


def find_optimal_inclusion_pct(
    final_signals: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/reversion_inclusion_thresholds.pkl",
    reoptimize: bool = False,
) -> Dict[str, float]:
    """
    Uses Optuna to optimize the inclusion/exclusion percentiles with persistent caching.

    Args:
        final_signals (pd.DataFrame): Weighted signal scores per ticker.
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of trials for Optuna optimization. Defaults to 50.
        n_jobs (int): Number of cores to use. Defaults to -1 (all cores)
        cache_filename (str): Storage path for caching.
        reoptimize (bool): Override to reoptimize.

    Returns:
        Dict[str, float]: A dictionary containing the best inclusion/exclusion percentiles.
    """
    # Load cached thresholds if available
    if reoptimize is False:
        cached_thresholds = load_parameters_from_pickle(cache_filename)
        if cached_thresholds:
            logger.info("Using cached thresholds.")
            return cached_thresholds

    # Create Optuna study using in-memory storage
    study = optuna.create_study(
        study_name="inclusion_thresholds",
        direction="maximize",
    )
    logger.info("Starting optimization...")

    # Run trials in parallel
    study.optimize(
        lambda trial: objective(trial, final_signals, returns_df),
        n_trials=n_trials,
        n_jobs=n_jobs,  # Use all available CPU cores
    )

    # Retrieve the best parameters
    best_params = study.best_trial.params if study.best_trial else {}

    optimal_thresholds = {
        "include_threshold_pct": best_params.get("include_threshold_pct", None),
        "exclude_threshold_pct": best_params.get("exclude_threshold_pct", None),
    }

    save_parameters_to_pickle(optimal_thresholds, cache_filename)
    return optimal_thresholds


def objective(trial, final_signals: pd.DataFrame, returns_df: pd.DataFrame) -> float:
    """
    Use Optuna to optimize the inclusion/exclusion thresholds while handling different stock history lengths.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        final_signals (pd.DataFrame): Weighted time-series signals per ticker.
        returns_df (pd.DataFrame): Log returns DataFrame.

    Returns:
        float: Cumulative return based on optimized thresholds.
    """
    # Ensure `final_signals` has a datetime index
    final_signals = final_signals.copy()
    final_signals.index = pd.to_datetime(final_signals.index)

    # Search space for inclusion/exclusion thresholds (percentiles)
    include_threshold_pct = trial.suggest_float(
        "include_threshold_pct", 0.1, 0.4, step=0.05
    )
    exclude_threshold_pct = trial.suggest_float(
        "exclude_threshold_pct", 0.1, 0.4, step=0.05
    )

    # Convert DataFrames to NumPy arrays for performance
    dates = returns_df.index.to_numpy()
    returns_data = returns_df.to_numpy()
    signal_data = final_signals.reindex(returns_df.index).to_numpy()

    positions = np.zeros_like(returns_data, dtype=float)

    def process_date(idx):
        if np.isnan(signal_data[idx]).all():
            return np.zeros(returns_data.shape[1])  # No signals for this date

        include_threshold = np.nanquantile(signal_data[idx], 1 - include_threshold_pct)
        exclude_threshold = np.nanquantile(signal_data[idx], exclude_threshold_pct)

        include_mask = signal_data[idx] >= include_threshold
        exclude_mask = signal_data[idx] <= exclude_threshold

        include_mask &= ~exclude_mask  # Ensure no overlap
        exclude_mask &= ~include_mask

        result = np.zeros(returns_data.shape[1], dtype=float)
        result[include_mask] = 1
        result[exclude_mask] = -1
        return result

    # Parallelize processing across dates
    results = Parallel(n_jobs=-1)(delayed(process_date)(i) for i in range(len(dates)))
    positions[:] = np.array(results)

    # Simulate strategy
    _, cumulative_return = simulate_strategy(returns_df, positions)
    return cumulative_return


def simulate_strategy(
    returns_df: pd.DataFrame, positions: np.ndarray
) -> Tuple[pd.Series, float]:
    """
    Simulates the strategy using positions and calculates cumulative return.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        positions (np.ndarray): Positions numpy array.

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - strategy_returns (pd.Series): Daily returns of the strategy.
            - cumulative_return (float): Final cumulative return of the strategy.
    """
    strategy_returns = np.sum(
        np.roll(positions, shift=1, axis=0) * returns_df.to_numpy(), axis=1
    )
    cumulative_return = (
        np.prod(strategy_returns + 1) - 1
    )  # More accurate cumulative return
    return pd.Series(strategy_returns, index=returns_df.index), cumulative_return
