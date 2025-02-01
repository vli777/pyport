from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from reversion.multiscale_reversion import calculate_robust_z_scores
from utils.performance_metrics import sharpe_ratio
from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def optimize_robust_mean_reversion(
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/reversion_window_multiplier.pkl",
    reoptimize: bool = False,
) -> Tuple[Dict[str, float], optuna.study.Study]:
    """
    Optimize the rolling window and z_threshold using Optuna.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        window_range (range): Range of window sizes.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        cache_filename (str, optional): Path to Pickle cache file.
        reoptimize (bool): Override to reoptimize.

    Returns:
        best parameters and the study object.
    """
    # Load cached study results if available and not reoptimizing
    if not reoptimize:
        cached_params = load_parameters_from_pickle(cache_filename)
        if cached_params:
            logger.info("Using cached parameters.")
            return cached_params, None

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: robust_mean_reversion_objective(trial, returns_df),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    best_params = (
        study.best_trial.params
        if study.best_trial
        else {"window": 20, "z_threshold": 1.5}
    )
    save_parameters_to_pickle(best_params, cache_filename)
    return best_params, study


def robust_mean_reversion_objective(trial, returns_df: pd.DataFrame) -> float:
    """
    Objective function for optimizing the robust mean reversion parameters.
    The trial suggests a rolling window size and a z_threshold.
    The resulting signals are used (with a one-day shift to avoid lookahead)
    to simulate a simple strategy; the cumulative return and sharpe ratio is maximized.
    """
    window = trial.suggest_int("window", 10, 30, step=5)
    z_threshold = trial.suggest_float("z_threshold", 1.0, 3.0, step=0.1)

    robust_z = calculate_robust_z_scores(returns_df, window)
    # Generate signals on all tickers at once; result is a DataFrame of {date x ticker}
    signals_df = robust_z.map(
        lambda x: 1 if x < -z_threshold else (-1 if x > z_threshold else 0)
    )

    # Shift signals to avoid lookahead bias
    positions_df = signals_df.shift(1).fillna(0)

    # Limit analysis to tickers with sufficient data
    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    positions_df = positions_df[valid_stocks]
    aligned_returns = returns_df[valid_stocks].reindex(positions_df.index)

    # Calculate Sharpe Ratio
    sharpe = sharpe_ratio(aligned_returns)

    _, cumulative_return = simulate_strategy(aligned_returns, positions_df)

    composite_score = 0.5 * (sharpe + cumulative_return)

    return composite_score


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
