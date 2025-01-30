from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def optimize_mean_reversion(
    returns_df: pd.DataFrame,
    window_range=range(5, 31, 5),
    n_trials: int = 50,
    n_jobs: int = -1,  # All cores
    cache_filename: str = "optuna_cache/reversion_thresholds.pkl",
    reoptimize: bool = False,
) -> dict:
    """
    Optimize mean reversion strategy using Optuna and cache results.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        window_range (range): Range of window sizes.
        n_trials (int, optional): Number of optimization trials. Defaults to 100.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        cache_filename (str, optional): Path to Pickle cache file.
        reoptimize (bool): Override to reoptimize.

    Returns:
        dict: Optimized thresholds per ticker.
    """
    # Load cached study results if available and not reoptimizing
    cached_params = None
    if not reoptimize:
        cached_params = load_parameters_from_pickle(cache_filename)
        if cached_params:
            logger.info("Using cached parameters.")
            return cached_params, None

    # Create Optuna study (without SQLite)
    study = optuna.create_study(
        study_name="mean_reversion_thresholds",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=max(5, n_trials // 10)),
    )

    # Precompute rolling standard deviations for all window sizes
    rolling_std_cache = {
        w: returns_df.rolling(window=w, min_periods=1).std().replace(0, np.nan)
        for w in window_range
    }

    # Run trials in parallel
    study.optimize(
        lambda trial: objective(trial, window_range, returns_df, rolling_std_cache),
        n_trials=n_trials,
        n_jobs=n_jobs,  # Use all available CPU cores
    )

    # Save best parameters to cache
    save_parameters_to_pickle(study, cache_filename)

    # Compute dynamic thresholds using optimized parameters
    best_params = study.best_params
    dynamic_thresholds = scale_thresholds_per_ticker(
        returns_df=returns_df,
        base_overbought_multiplier=best_params["overbought_multiplier"],
        base_oversold_multiplier=best_params["oversold_multiplier"],
        rolling_std_cache=rolling_std_cache[best_params["window"]],
        window=best_params["window"],
    )

    return dynamic_thresholds.study


def objective(
    trial, window_range: range, returns_df: pd.DataFrame, rolling_std_cache: dict
) -> float:
    """
    Objective function for Optuna to optimize mean reversion strategy.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        window_range (range): Range of windows to test.
        returns_df (pd.DataFrame): Log returns DataFrame.
        rolling_std_cache (dict): Precomputed rolling standard deviations.

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

    # Compute rolling mean only for the selected window
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


def scale_thresholds_per_ticker(
    returns_df: pd.DataFrame,
    base_overbought_multiplier: float,
    base_oversold_multiplier: float,
    rolling_std_cache: Dict[int, pd.DataFrame],
    window: int,
    alpha: float = 0.42,
) -> Dict[str, Tuple[float, float]]:
    """
    Hybrid approach: Uses historical rolling std but dynamically adjusts for recent volatility.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        base_overbought_multiplier (float): Optimized global overbought Z-score multiplier.
        base_oversold_multiplier (float): Optimized global oversold Z-score multiplier.
        rolling_std_cache (dict): Precomputed rolling std for different windows.
        window (int): Optimized rolling window size.
        alpha (float): Weight for historical std (higher = more stable, lower = more reactive).

    Returns:
        dict: Adjusted thresholds per ticker.
    """
    # Use precomputed rolling standard deviation (historical)
    historical_rolling_std = rolling_std_cache[window]

    # Compute recent volatility using an exponentially weighted moving standard deviation (EWMSD)
    recent_rolling_std = returns_df.ewm(span=window, adjust=False).std()

    # Blend historical and recent volatility based on alpha weighting
    blended_std = alpha * historical_rolling_std + (1 - alpha) * recent_rolling_std

    # Compute per-ticker deviations
    std_dev_per_ticker = blended_std.iloc[
        -1
    ]  # Use latest period's blended standard deviation
    global_mean_std = std_dev_per_ticker.mean()  # Average std across all tickers

    # Scale thresholds dynamically
    scaled_thresholds = {
        ticker: (
            base_overbought_multiplier
            * (1 + std_dev_per_ticker[ticker] / global_mean_std),
            base_oversold_multiplier
            * (1 + std_dev_per_ticker[ticker] / global_mean_std),
        )
        for ticker in returns_df.columns
    }
    return scaled_thresholds


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
