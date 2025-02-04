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
    group_id: str = None,
) -> Dict[str, float]:
    """
    Run Optuna to find the optimal weighting of daily and weekly signals.
    Returns a dictionary with both weight_daily and weight_weekly, ensuring they sum to 1.
    """
    study = optuna.create_study(
        study_name=f"reversion_weights_optimization_{group_id}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=max(5, n_trials // 10)),
    )
    study.optimize(
        lambda trial: reversion_weights_objective(trial, daily_signals_df, weekly_signals_df, returns_df),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    if study.best_trial is None:
        logger.error("No valid optimization results found.")
        best_weights = {"weight_daily": 0.7, "weight_weekly": 0.3}
    else:
        best_weights = study.best_trial.params        
        best_weights["weight_weekly"] = round(1.0 - best_weights["weight_daily"], 1)
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
    combined_signals = weight_daily * daily_signals_df + weight_weekly * weekly_signals_df

    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    combined_signals = combined_signals[valid_stocks]

    aligned_returns = returns_df[valid_stocks].reindex(combined_signals.index)
    positions_df = combined_signals.shift(1).fillna(0)

    # Run the simulation and calculate a composite score.
    _, metrics = simulate_strategy(aligned_returns, positions_df)

    return composite_score(metrics)
