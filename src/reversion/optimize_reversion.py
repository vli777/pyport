from typing import List, Tuple
import numpy as np
import optuna
import pandas as pd


def objective(trial, window_range: range, returns_df: pd.DataFrame) -> float:
    """
    Objective function for Optuna to optimize mean reversion strategy.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        window_range (range): Range of windows to test.
        returns_df (pd.DataFrame): Log returns DataFrame.

    Returns:
        float: Cumulative return.
    """
    # Extract min, max, and step from the range object
    window_min, window_max, window_step = (
        window_range.start,
        window_range.stop - 1,
        window_range.step,
    )

    # Define hyperparameter search space
    window = trial.suggest_int("window", window_min, window_max, step=window_step)
    overbought_multiplier = trial.suggest_float(
        "overbought_multiplier", 1.5, 2.5, step=0.1
    )
    oversold_multiplier = trial.suggest_float(
        "oversold_multiplier", -2.5, -1.5, step=0.1
    )

    # Calculate Z-scores
    rolling_mean = returns_df.rolling(window=window, min_periods=1).mean()
    rolling_std = returns_df.rolling(window=window, min_periods=1).std()
    z_score_df = (returns_df - rolling_mean) / rolling_std.replace(0, np.nan)

    # Define dynamic thresholds
    dynamic_thresholds = {
        ticker: (
            overbought_multiplier * z_score_df[ticker].std(),
            oversold_multiplier * z_score_df[ticker].std(),
        )
        for ticker in returns_df.columns
    }

    # Simulate strategy performance
    _, cumulative_return = simulate_strategy(returns_df, dynamic_thresholds, z_score_df)

    return cumulative_return


def optimize_mean_reversion(
    returns_df: pd.DataFrame,
    window_range=range(5, 31, 5),
    n_trials: int = 100,
    n_jobs: int = -1,
) -> optuna.Study:
    """
    Optimize mean reversion strategy using Optuna.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of optimization trials. Defaults to 100.

    Returns:
        optuna.Study: The optimization study.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial=trial, window_range=window_range, returns_df=returns_df
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    return study


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
