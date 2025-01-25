import numpy as np
import optuna
import pandas as pd

from signals.dynamic_threshold import get_dynamic_thresholds
from signals.z_score import calculate_z_score


def optimize_multiplier(returns_df, window=20, n_trials=50):
    """
    Optimize the multiplier using Optuna to maximize cumulative return.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        window (int): Rolling window size for Z-score calculation.
        n_trials (int): Number of optimization trials.

    Returns:
        float: Best multiplier value.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, returns_df, window), n_trials=n_trials
    )
    return study.best_params["multiplier"], study.best_value


def objective(trial, returns_df, window=20):
    """
    Optuna objective function to find the optimal multiplier.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        returns_df (pd.DataFrame): Log returns DataFrame.
        window (int): Rolling window size for Z-score calculation.

    Returns:
        float: Objective value (e.g., cumulative return).
    """
    # Suggest multiplier within a reasonable range
    multiplier = trial.suggest_float("multiplier", 1.0, 2.0, step=0.1)

    # Calculate Z-scores
    z_scores_df = calculate_z_score(returns_df, window)

    # Calculate dynamic thresholds
    dynamic_thresholds = get_dynamic_thresholds(
        returns_df, window=window, multiplier=multiplier
    )

    # Simulate the strategy and calculate performance
    cumulative_return = simulate_strategy(returns_df, dynamic_thresholds, z_scores_df)

    return cumulative_return


def simulate_strategy(returns_df, dynamic_thresholds, z_scores_df):
    """
    Simulates the strategy using thresholds and calculates cumulative return.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        dynamic_thresholds (dict): Thresholds for overbought/oversold conditions.
        z_scores_df (pd.DataFrame): Precomputed Z-scores DataFrame.

    Returns:
        float: Simulated cumulative return.
    """
    positions = pd.DataFrame(0, index=returns_df.index, columns=returns_df.columns)

    # Generate signals based on thresholds
    for ticker in returns_df.columns:
        overbought, oversold = dynamic_thresholds[ticker]
        z_scores = z_scores_df[ticker]

        # Buy when oversold, sell when overbought
        positions[ticker] = np.where(
            z_scores < oversold,
            1,  # Long position
            np.where(z_scores > overbought, -1, 0),  # Short position
        )

    # Calculate daily returns from positions
    strategy_returns = (positions.shift(1) * returns_df).sum(axis=1)
    cumulative_return = np.exp(strategy_returns.cumsum())[-1]  # Final cumulative return

    return cumulative_return
