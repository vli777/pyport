from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from reversion.optimize_reversion import optimize_mean_reversion
from reversion.zscore_plot import plot_z_scores_grid
from utils import logger
from utils.portfolio_utils import resample_returns


def apply_mean_reversion_multiscale(
    returns_df: pd.DataFrame, plot: bool = False, n_jobs: int = -1, n_trials: int = 50
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Apply mean reversion strategy across multiple time scales with dynamic rolling windows.

    Args:
        returns_df (pd.DataFrame): Daily log returns DataFrame.
        plot (bool): Whether to plot Z-Scores.
        n_jobs (int): Number of parallel jobs.
        n_trials (int): Number of optimization trials.

    Returns:
        Dict[str, Dict[str, List[str]]:
            - Signals structured by time scale.
    """
    resampled = resample_returns(returns_df)  # Assume resampling function exists
    signals = {}

    for timeframe, returns in [("daily", returns_df), ("weekly", resampled["weekly"])]:
        logger.info(f"Optimizing for {timeframe} data...")

        # Optimize parameters using modular function
        study = optimize_mean_reversion(
            returns_df=returns,
            window_range=range(5, 31, 5) if timeframe == "daily" else range(2, 9),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        optimized_params = study.best_trial.params
        logger.info(
            f"{timeframe.capitalize()} Optimized parameters: {optimized_params}"
        )

        # Calculate Z-scores and thresholds
        window = optimized_params.get("window", 20)
        overbought_multiplier = optimized_params.get("overbought_multiplier", 2.0)
        oversold_multiplier = optimized_params.get("oversold_multiplier", -2.0)

        z_score_df = calculate_z_scores(returns, window)
        dynamic_thresholds = get_dynamic_thresholds(
            z_score_df, overbought_multiplier, oversold_multiplier
        )

        if plot:
            plot_z_scores_grid(z_score_df, dynamic_thresholds)

        # Generate signals
        signals[timeframe] = reversion_signals_filter(z_score_df, dynamic_thresholds)

    return signals


def calculate_z_scores(returns_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate rolling Z-scores for given return data and window size."""
    rolling_mean = returns_df.rolling(window=window, min_periods=1).mean()
    rolling_std = returns_df.rolling(window=window, min_periods=1).std()
    return (returns_df - rolling_mean) / rolling_std.replace(0, np.nan)


def get_dynamic_thresholds(
    z_score_df: pd.DataFrame, overbought_multiplier: float, oversold_multiplier: float
) -> Dict[str, Tuple[float, float]]:
    """Compute dynamic overbought/oversold thresholds for each ticker."""
    return {
        ticker: (
            overbought_multiplier * z_score_df[ticker].std(),
            oversold_multiplier * z_score_df[ticker].std(),
        )
        for ticker in z_score_df.columns
    }


def reversion_signals_filter(
    z_score_df: pd.DataFrame, dynamic_thresholds: Dict[str, Tuple[float, float]]
) -> Dict[str, Dict[str, int]]:
    """
    Generate mean reversion signals based on Z-Score and dynamic thresholds.

    Args:
        z_score_df (pd.DataFrame): Z-Score DataFrame with tickers as columns and dates as index.
        dynamic_thresholds (dict): {ticker: (overbought_threshold, oversold_threshold)}

    Returns:
        dict: {ticker: {date: signal}}
    """
    signals = {}

    for ticker in z_score_df.columns:
        signals[ticker] = (
            (z_score_df[ticker] < dynamic_thresholds[ticker][1]).astype(int)  # Buy
            - (z_score_df[ticker] > dynamic_thresholds[ticker][0]).astype(int)  # Sell
        ).to_dict()  # Convert to dict with date as key

    return signals
