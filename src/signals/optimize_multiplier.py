import numpy as np
import optuna
import pandas as pd

from signals.dynamic_threshold import get_dynamic_thresholds
from signals.z_score import calculate_z_score


def optimize_multiplier(returns_df, window=20, n_trials=50):
    """
    Optimize asymmetric multipliers for overbought and oversold thresholds using Optuna.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        window (int): Rolling window size for Z-score calculation.
        n_trials (int): Number of optimization trials.

    Returns:
        dict: A dictionary containing the best parameters and associated statistics.
              Example: {"overbought_multiplier": 1.2, "oversold_multiplier": 2.5, "objective_value": 1.25}
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, returns_df, window), n_trials=n_trials
    )

    # Fetch best parameters and performance
    best_params = study.best_params
    best_value = study.best_value

    best_params["objective_value"] = best_value
    return best_params


def objective(trial, returns_df, window=20):
    """
    Optuna objective function to optimize thresholds and rolling window.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        returns_df (pd.DataFrame): Log returns DataFrame with tickers as columns.

    Returns:
        float: Objective value (e.g., risk-adjusted cumulative return).
    """
    # Suggest multiplier and rolling window size
    overbought_multiplier = trial.suggest_float(
        "overbought_multiplier", 1.0, 2.2, step=0.1
    )
    oversold_multiplier = trial.suggest_float("oversold_multiplier", 1.0, 2.2, step=0.1)
    window = trial.suggest_int("window", 5, 30, step=5)

    # Calculate Z-scores
    z_scores_df = calculate_z_score(returns_df, window)

    # Calculate dynamic thresholds
    dynamic_thresholds = get_dynamic_thresholds(
        returns_df,
        window=window,
        overbought_multiplier=overbought_multiplier,
        oversold_multiplier=oversold_multiplier,
    )

    # Simulate the strategy
    strategy_returns, cumulative_return = simulate_strategy(
        returns_df, dynamic_thresholds, z_scores_df
    )

    # Check for invalid returns
    if cumulative_return <= 0 or np.isnan(cumulative_return):
        return float("-inf")  # Penalize invalid results

    # Calculate Sharpe ratio safely
    if strategy_returns.std() == 0 or np.isnan(strategy_returns.std()):
        return float("-inf")  # Penalize invalid Sharpe ratio

    sharpe_ratio = strategy_returns.mean() / strategy_returns.std()

    objective_value = sharpe_ratio * cumulative_return

    return objective_value


def simulate_strategy(returns_df, dynamic_thresholds, z_scores_df):
    """
    Simulates the strategy using thresholds and calculates cumulative return.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        dynamic_thresholds (dict): Thresholds for overbought/oversold conditions.
        z_scores_df (pd.DataFrame): Precomputed Z-scores DataFrame.

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - strategy_returns (pd.Series): Daily returns of the strategy.
            - cumulative_return (float): Final cumulative return of the strategy.
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

    # Calculate daily strategy returns
    strategy_returns = (positions.shift(1) * returns_df).sum(axis=1)

    # Calculate cumulative return
    cumulative_return = np.exp(strategy_returns.cumsum().iloc[-1])

    return strategy_returns, cumulative_return
