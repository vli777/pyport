from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from reversion.strategy_metrics import composite_score, simulate_strategy
from utils.logger import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def optimize_group_weights(
    group_signals: dict,
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    reoptimize: bool = False,
):
    group_weights = {}
    # group_signals is expected to be a dict keyed by group label,
    # where each value is a dict:
    # {
    #    "tickers": [list of tickers],
    #    "daily": {ticker: {date: signal}, ...},
    #    "weekly": {ticker: {date: signal}, ...},
    # }
    for group_label, data in group_signals.items():
        tickers = data["tickers"]
        group_returns = returns_df[tickers]
        cache_filename = f"optuna_cache/reversion_period_weights_{group_label}.pkl"

        # Precompute the signal dataframes once for the group
        daily_signals_df = pd.DataFrame.from_dict(
            {t: data["daily"][t] for t in tickers if t in data["daily"]},
            orient="index",
        ).T
        weekly_signals_df = pd.DataFrame.from_dict(
            {t: data["weekly"][t] for t in tickers if t in data["weekly"]},
            orient="index",
        ).T

        # Create a combined index outside the trial loop to avoid repeated reindexing
        combined_dates = daily_signals_df.index.union(weekly_signals_df.index).union(
            group_returns.index
        )
        daily_signals_df = daily_signals_df.reindex(combined_dates).fillna(0)
        weekly_signals_df = weekly_signals_df.reindex(combined_dates).fillna(0)

        optimal_weights = find_optimal_weights(
            daily_signals_df,
            weekly_signals_df,
            group_returns,
            n_trials=n_trials,
            n_jobs=n_jobs,
            cache_filename=cache_filename,
            reoptimize=reoptimize,
        )
        group_weights[group_label] = optimal_weights
    return group_weights


def find_optimal_weights(
    daily_signals_df: pd.DataFrame,
    weekly_signals_df: pd.DataFrame,
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
        daily_signals_df (pd.DataFrame): Daily signals DataFrame.
        weekly_signals_df (pd.DataFrame): Weekly signals DataFrame.
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
    )

    # Run optimization in parallel
    study.optimize(
        lambda trial: reversion_weights_objective(
            trial, daily_signals_df, weekly_signals_df, returns_df
        ),
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
