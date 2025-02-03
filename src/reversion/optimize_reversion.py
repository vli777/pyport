from typing import Dict, List, Tuple
import optuna
import pandas as pd

from reversion.strategy_metrics import composite_score, simulate_strategy
from reversion.z_scores import calculate_robust_z_scores
from utils.logger import logger
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

    _, metrics = simulate_strategy(aligned_returns, positions_df)

    return composite_score(metrics)
