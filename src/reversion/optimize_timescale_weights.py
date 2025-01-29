from typing import Dict, List, Tuple
import optuna
import pandas as pd


def find_optimal_weights(
    reversion_signals: Dict[str, Dict[str, Dict[str, int]]],
    returns_df: pd.DataFrame,
    n_trials: int = 50,
) -> Dict[str, float]:
    """
    Run Optuna to find the optimal weighting of daily and weekly signals.

    Args:
        reversion_signals (Dict[str, Dict[str, Dict[str, int]]]): Dictionary of signals per timeframe.
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.

    Returns:
        Dict[str, float]: Best weights for daily and weekly signals.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optimize_signal_weights(trial, reversion_signals, returns_df),
        n_trials=n_trials,
        n_jobs=-1,  # Utilize all cores
    )
    # Returns optimal { "weight_daily": x, "weight_weekly": y }
    return study.best_trial.params


def optimize_signal_weights(trial, reversion_signals, returns_df) -> float:
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
    weight_daily = trial.suggest_float("weight_daily", 0.0, 1.0, step=0.05)
    weight_weekly = 1.0 - weight_daily  # Ensuring sum = 1.0 for interpretability

    # Convert dictionary signals to DataFrames with dates as index and tickers as columns
    daily_signals = pd.DataFrame.from_dict(reversion_signals["daily"], orient="index").T
    weekly_signals = pd.DataFrame.from_dict(
        reversion_signals["weekly"], orient="index"
    ).T

    # Ensure both DataFrames have the same index (dates)
    combined_dates = daily_signals.index.union(weekly_signals.index)
    daily_signals = daily_signals.reindex(combined_dates).fillna(0)
    weekly_signals = weekly_signals.reindex(combined_dates).fillna(0)

    # Weighted combination of signals
    combined_signals = weight_daily * daily_signals + weight_weekly * weekly_signals

    # Ensure signals are integers (-1, 0, 1) after weighting
    combined_signals = combined_signals.applymap(
        lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
    )

    # Simulate strategy with weighted signals
    _, cumulative_return = simulate_strategy(returns_df, combined_signals)

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
