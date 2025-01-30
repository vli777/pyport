from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from utils.performance_metrics import sharpe_ratio
from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def optimize_mean_reversion(
    returns_df: pd.DataFrame,
    window_range=range(5, 31, 5),
    n_trials: int = 50,
    n_jobs: int = -1,  # All cores
    cache_filename: str = "optuna_cache/reversion_window_multiplier.pkl",
    reoptimize: bool = False,
) -> dict:
    """
    Optimize mean reversion strategy using Optuna and cache results.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        window_range (range): Range of window sizes.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        cache_filename (str, optional): Path to Pickle cache file.
        reoptimize (bool): Override to reoptimize.

    Returns:
        dict: Optimized parameters.
    """
    # Load cached study results if available and not reoptimizing
    if not reoptimize:
        cached_params = load_parameters_from_pickle(cache_filename)
        if cached_params:
            logger.info("Using cached parameters.")
            return cached_params, None

    # Create Optuna study
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
        n_jobs=n_jobs,
    )

    # Save best parameters to cache
    if study.best_trial:
        best_params = study.best_trial.params
        save_parameters_to_pickle(best_params, cache_filename)
    else:
        logger.error("No valid optimization results found.")
        return None, study  # Avoid returning an invalid best_params

    return best_params, study


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
        float: Cumulative return or other performance metric.
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

    logger.debug(
        f"Trial {trial.number}: window={window}, overbought_multiplier={overbought_multiplier}, oversold_multiplier={oversold_multiplier}"
    )

    # Fetch precomputed rolling standard deviation with error handling
    rolling_std = rolling_std_cache.get(window)
    if rolling_std is None:
        logger.warning(
            f"Window size {window} not found in rolling_std_cache. Skipping trial."
        )
        return -np.inf  # Assign a poor score to invalidate this trial

    # Compute rolling mean only for the selected window
    rolling_mean = returns_df.rolling(window=window, min_periods=1).mean()

    # Calculate Z-scores dynamically based on available history
    z_score_df = (returns_df - rolling_mean) / rolling_std

    # Ensure only stocks with sufficient non-NaN data are considered
    min_non_na = int(0.5 * len(returns_df))  # Example: at least 50% non-NaN
    valid_stocks = returns_df.dropna(axis=1, thresh=min_non_na).columns
    z_score_df = z_score_df[valid_stocks]
    returns_df_filtered = returns_df[valid_stocks]

    if z_score_df.empty:
        logger.warning("No valid stocks after filtering. Skipping trial.")
        return -np.inf

    # Define dynamic thresholds per stock
    dynamic_thresholds = {
        ticker: (
            overbought_multiplier * z_score_df[ticker].std(skipna=True),
            oversold_multiplier * z_score_df[ticker].std(skipna=True),
        )
        for ticker in valid_stocks
    }

    # Simulate strategy performance
    strategy_returns, cumulative_return = simulate_strategy(
        returns_df_filtered, dynamic_thresholds, z_score_df
    )
    # Calculate Sharpe Ratio
    sharpe = sharpe_ratio(strategy_returns)

    logger.debug(
        f"Trial {trial.number}: Cumulative Return={cumulative_return} Sharpe Ratio={sharpe}"
    )

    return 0.5 * (sharpe + cumulative_return)


def simulate_strategy(
    returns_df: pd.DataFrame,
    dynamic_thresholds: Dict[str, Tuple[float, float]],
    z_scores_df: pd.DataFrame,
) -> Tuple[pd.Series, float]:
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
    # Initialize positions DataFrame with zeros
    positions = pd.DataFrame(0, index=returns_df.index, columns=returns_df.columns)

    # Vectorized position assignment
    for ticker in dynamic_thresholds:
        overbought, oversold = dynamic_thresholds[ticker]
        z_scores = z_scores_df[ticker]

        # Assign positions based on thresholds
        positions[ticker] = np.where(
            z_scores < oversold, 1, np.where(z_scores > overbought, -1, 0)
        )

    # Ensure no NaN values in positions
    positions.fillna(0, inplace=True)

    # Shift positions to prevent look-ahead bias
    shifted_positions = positions.shift(1).fillna(0)

    # Calculate daily strategy returns
    strategy_returns = (shifted_positions * returns_df).sum(axis=1)

    # Calculate cumulative return
    cumulative_return = np.exp(strategy_returns.cumsum().iloc[-1]) - 1

    return strategy_returns, cumulative_return


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
