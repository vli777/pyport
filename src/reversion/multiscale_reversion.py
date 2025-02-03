from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from reversion.optimize_reversion import optimize_robust_mean_reversion
from utils.z_scores import calculate_robust_zscores, plot_robust_z_scores
from utils.logger import logger


def apply_mean_reversion_multiscale(
    returns_df: pd.DataFrame, plot: bool = False, n_jobs: int = -1, n_trials: int = 50
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Apply a robust mean reversion strategy over multiple timeframes.
    For each timeframe (daily and weekly), optimize the rolling window and
    z_threshold, compute robust z-scores, and then generate buy/sell signals.

    Args:
        returns_df (pd.DataFrame): Daily log returns DataFrame.
        plot (bool): Whether to plot Z-Scores.
        n_jobs (int): Number of parallel jobs.
        n_trials (int): Number of optimization trials.

    Returns:
        Dict[str, Dict[str, Dict[str, int]]]:
        Dict with keys 'daily' and 'weekly', each mapping tickers to a {date: signal} dict.
    """
    signals = {}

    # Daily signals
    logger.info("Optimizing robust mean reversion for daily data...")
    best_params_daily, _ = optimize_robust_mean_reversion(
        returns_df, n_trials=n_trials, n_jobs=n_jobs
    )
    window_daily = best_params_daily.get("window", 20)
    z_threshold_daily = best_params_daily.get("z_threshold", 1.5)
    robust_z_daily = calculate_robust_zscores(returns_df, window_daily)

    if plot:
        plot_robust_z_scores(robust_z_daily, z_threshold_daily)
    signals["daily"] = generate_reversion_signals(robust_z_daily, z_threshold_daily)

    # Weekly signals: resample returns (using last observation of the week)
    weekly_returns = returns_df.resample("W").last()
    logger.info("Optimizing robust mean reversion for weekly data...")
    best_params_weekly, _ = optimize_robust_mean_reversion(
        weekly_returns, n_trials=n_trials, n_jobs=n_jobs
    )
    window_weekly = best_params_weekly.get("window", 4)
    z_threshold_weekly = best_params_weekly.get("z_threshold", 1.5)
    robust_z_weekly = calculate_robust_zscores(weekly_returns, window_weekly)

    if plot:
        plot_robust_z_scores(robust_z_weekly, z_threshold_weekly)

    signals["weekly"] = generate_reversion_signals(robust_z_weekly, z_threshold_weekly)

    return signals


def generate_reversion_signals(
    robust_z: pd.DataFrame, z_threshold: float
) -> Dict[str, Dict[str, int]]:
    """
    Generate buy/sell signals based on robust z-scores.
    For each ticker, assign:
      - +1 (buy) if z-score < -z_threshold (oversold)
      - -1 (sell) if z-score > z_threshold (overbought)
      -  0 otherwise
    Returns a dict mapping tickers to a dict of {date: signal}.
    """
    signals = {}
    for ticker in robust_z.columns:
        signal_series = robust_z[ticker].apply(
            lambda x: 1 if x < -z_threshold else (-1 if x > z_threshold else 0)
        )
        signals[ticker] = signal_series.dropna().to_dict()
    return signals
