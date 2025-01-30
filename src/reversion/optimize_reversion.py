from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd


def optimize_mean_reversion(
    returns_df: pd.DataFrame,
    window_range=range(5, 31, 5),
    n_trials: int = 50,
    n_jobs: int = 1,  # Set to 1 to avoid SQLite locking issues, to-do: switch to psql
    storage_path: str = "optuna_cache/mean_reversion.db",
    study_name: str = "mean_reversion_thresholds",
) -> dict:
    """
    Optimize mean reversion strategy using Optuna and cache results.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        window_range (range): Range of window sizes.
        n_trials (int, optional): Number of optimization trials. Defaults to 100.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        storage_path (str, optional): Path to Optuna SQLite cache.
        study_name (str, optional): Name of the Optuna study.

    Returns:
        dict: Optimized thresholds for each ticker.
    """
    # Ensure the cache directory exists
    cache_dir = Path(storage_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load or create an Optuna study with SQLite storage
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",  # Use SQLite for persistence
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=max(5, n_trials // 10)
        ),  # Dynamic startup trials
        load_if_exists=True,  # Load existing study instead of restarting
    )

    # Determine the number of additional trials needed
    remaining_trials = n_trials - len(study.trials)
    if remaining_trials > 0:
        # Compute rolling std in advance for all window sizes to avoid redundant computations
        rolling_std_cache = {
            w: returns_df.rolling(window=w, min_periods=1).std().replace(0, np.nan)
            for w in window_range
        }

        # Run optimization
        study.optimize(
            lambda trial: objective(trial, window_range, returns_df, rolling_std_cache),
            n_trials=remaining_trials,
            n_jobs=n_jobs,
        )

    # Extract best parameters
    best_params = study.best_params
    window = best_params["window"]
    overbought_multiplier = best_params["overbought_multiplier"]
    oversold_multiplier = best_params["oversold_multiplier"]

    # Compute new dynamic thresholds for each stock using cached rolling_std
    rolling_std = rolling_std_cache[window]

    dynamic_thresholds = {
        ticker: (
            overbought_multiplier * rolling_std[ticker].std(skipna=True),
            oversold_multiplier * rolling_std[ticker].std(skipna=True),
        )
        for ticker in returns_df.columns
    }

    return dynamic_thresholds, study


def objective(
    trial, window_range: range, returns_df: pd.DataFrame, rolling_std_cache: dict
) -> float:
    """
    Objective function for Optuna to optimize mean reversion strategy.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        window_range (range): Range of windows to test.
        returns_df (pd.DataFrame): Log returns DataFrame.
        rolling_std_cache (dict): Precomputed rolling standard deviations for efficiency.

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
        "overbought_multiplier", 1.5, 2.5, step=0.05
    )
    oversold_multiplier = trial.suggest_float(
        "oversold_multiplier", -2.5, -1.5, step=0.05
    )

    # Fetch precomputed rolling standard deviation
    rolling_std = rolling_std_cache[window]

    # Compute rolling mean only for the selected window (avoiding unnecessary recalculations)
    rolling_mean = returns_df.rolling(window=window, min_periods=1).mean()

    # Calculate Z-scores dynamically based on available history
    z_score_df = (returns_df - rolling_mean) / rolling_std

    # Ensure only stocks with valid history are considered
    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    z_score_df = z_score_df[valid_stocks]
    returns_df = returns_df[valid_stocks]

    # Define dynamic thresholds per stock
    dynamic_thresholds = {
        ticker: (
            overbought_multiplier * z_score_df[ticker].std(skipna=True),
            oversold_multiplier * z_score_df[ticker].std(skipna=True),
        )
        for ticker in valid_stocks
    }

    # Simulate strategy performance
    _, cumulative_return = simulate_strategy(returns_df, dynamic_thresholds, z_score_df)

    return cumulative_return


def simulate_strategy(
    returns_df: pd.DataFrame,
    dynamic_thresholds: Dict[str, Tuple[float, float]],
    z_scores_df: pd.DataFrame,
):
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

    for ticker in returns_df.columns:
        if ticker not in dynamic_thresholds:
            continue  # Skip tickers with insufficient data

        overbought, oversold = dynamic_thresholds[ticker]
        z_scores = z_scores_df[ticker]

        # Buy when oversold, sell when overbought
        positions[ticker] = np.where(
            z_scores < oversold,
            1,  # Long position
            np.where(z_scores > overbought, -1, 0),  # Short position
        )

    # Ensure stocks with missing data are handled properly
    positions.fillna(0, inplace=True)

    # Calculate daily strategy returns using shifted positions to prevent look-ahead bias
    strategy_returns = (positions.shift(1) * returns_df).sum(axis=1)

    # Calculate cumulative return (log to normal return conversion)
    cumulative_return = np.exp(strategy_returns.cumsum().iloc[-1]) - 1

    return strategy_returns, cumulative_return
