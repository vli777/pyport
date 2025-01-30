import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import optuna
import pandas as pd

from utils.logger import logger
from utils.caching_utils import (
    load_parameters_from_pickle,
    save_parameters_to_pickle,
)


def find_optimal_weights(
    reversion_signals: Dict[str, Dict[str, Dict[str, int]]],
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/reversion_period_weights.pkl",
    reoptimize: bool = False,
) -> Dict[str, float]:
    """
    Run Optuna to find the optimal weighting of daily and weekly signals.
    Uses built-in Optuna SQLite caching.

    Args:
        reversion_signals (Dict[str, Dict[str, Dict[str, int]]]): Dictionary of signals per timeframe.
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        cache_filename (str, optional): Path to Pickle cache file.
        reoptimize (bool): Override to reoptimize.

    Returns:
        Dict[str, float]: Best weights for daily and weekly signals.
    """
    # Load cached results if available and reoptimization is not forced
    if not reoptimize:
        cached_weights = load_parameters_from_pickle(cache_filename)
        if isinstance(cached_weights, dict): 
            return cached_weights

    # Load or create the study
    study = optuna.create_study(
        study_name="reversion_weights_optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=max(5, n_trials // 10)),
        load_if_exists=True,
    )

    # Run optimization in parallel
    study.optimize(
        lambda trial: objective(trial, reversion_signals, returns_df),
        n_trials=n_trials,
        n_jobs=n_jobs,        
    )

    if study.best_trial is None:
        logger.error("No valid optimization results found.")
        # Return default weights if optimization fails
        return {"weight_daily": 0.5, "weight_weekly": 0.5}

    # Extract best weights
    best_weights = study.best_trial.params

    # Save best weights (NOT the study object)
    save_parameters_to_pickle(best_weights, cache_filename)

    return best_weights


def objective(
    trial,
    reversion_signals: Dict[str, Dict[str, Dict[str, int]]],
    returns_df: pd.DataFrame,
) -> float:
    """
    Optimize the weights of different time scale signals for mean reversion.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        reversion_signals (Dict[str, Dict[str, Dict[str, int]]]): Dictionary of signals per timeframe.
        returns_df (pd.DataFrame): Log returns DataFrame.

    Returns:
        float: Cumulative return from the optimized strategy.
    """
    # Hyperparameter search space: Weight allocation to each timeframe
    weight_daily = trial.suggest_float("weight_daily", 0.0, 1.0, step=0.10)
    weight_weekly = 1.0 - weight_daily  # Ensuring sum = 1.0 for interpretability

    # Convert dictionary signals to DataFrames
    daily_signals = pd.DataFrame.from_dict(reversion_signals["daily"], orient="index").T
    weekly_signals = pd.DataFrame.from_dict(
        reversion_signals["weekly"], orient="index"
    ).T

    # Align both signals and returns DataFrames
    combined_dates = daily_signals.index.union(weekly_signals.index).union(
        returns_df.index
    )
    daily_signals = daily_signals.reindex(combined_dates).fillna(0)
    weekly_signals = weekly_signals.reindex(combined_dates).fillna(0)
    returns_df = returns_df.reindex(combined_dates)

    # Weighted combination of signals
    combined_signals: pd.DataFrame = weight_daily * daily_signals + weight_weekly * weekly_signals

    # Ensure signals remain in {-1, 0, 1}
    combined_signals = combined_signals.map(
        lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
    )
    
    # Ensure only stocks with valid history are considered
    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    combined_signals = combined_signals[valid_stocks]
    returns_df = returns_df[valid_stocks]

    # Convert signals to positions while ensuring no look-ahead bias
    positions_df = combined_signals.shift(1)  # Shift positions to avoid look-ahead bias
    positions_df.fillna(0, inplace=True)  # Fill missing positions with 0 (neutral)
    
    # Simulate strategy with valid positions
    _, cumulative_return = simulate_strategy(returns_df, positions_df=positions_df)

    return cumulative_return  # Optuna maximizes this


def simulate_strategy(
    returns_df: pd.DataFrame, positions_df: pd.DataFrame
) -> Tuple[pd.Series, float]:
    """
    Simulates the strategy using positions and calculates cumulative return.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        positions_df (pd.DataFrame): Positions DataFrame with tickers as columns and dates as index.

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - strategy_returns (pd.Series): Daily returns of the strategy.
            - cumulative_return (float): Final cumulative return of the strategy.
    """
    # Calculate daily strategy returns using previous day's positions to avoid look-ahead bias
    strategy_returns = (positions_df.shift(1) * returns_df).sum(axis=1)

    # Calculate cumulative return
    cumulative_return = (
        strategy_returns + 1
    ).prod() - 1  # More accurate cumulative return

    return strategy_returns, cumulative_return
