from typing import Callable, Dict, Optional, Tuple
import optuna
import pandas as pd
import numpy as np
import functools
import copy
from pathlib import Path
import os

from correlation.decorrelation import filter_correlated_groups
from utils.performance_metrics import calculate_portfolio_alpha
from utils.logger import logger


def optimize_correlation_threshold(
    returns_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float,
    sharpe_threshold: float = 0.005,
    linkage_method: str = "average",
    n_trials: int = 50,
    direction: str = "maximize",
    sampler: Optional[Callable] = None,
    pruner: Optional[Callable] = None,
    cache_dir: str = "optuna_cache/correlation_thresholds",
    cache_file: str = "correlation_study.db",
) -> Tuple[Dict[str, float], float]:
    """
    Optimize the correlation_threshold and lambda_weight using Optuna.
    Caches results in an Optuna SQLite database to avoid redundant optimizations.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns of tickers.
        performance_df (pd.DataFrame): DataFrame containing performance metrics.
        market_returns (pd.Series): Series containing market returns.
        risk_free_rate (float): Risk-free rate for alpha calculation.
        sharpe_threshold (float, optional): Threshold for Sharpe Ratio filtering.
        linkage_method (str, optional): Method for hierarchical clustering.
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        direction (str, optional): Optimization direction ("maximize" or "minimize").
        sampler (Callable, optional): Optuna sampler.
        pruner (Callable, optional): Optuna pruner.
        cache_dir (str): Directory for caching Optuna studies.
        cache_file (str): Cache database filename.

    Returns:
        Tuple[Dict[str, float], float]: Best parameters and best objective value.
    """
    # Ensure cache directory exists
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Define full storage path with SQLite URL format
    storage_url = f"sqlite:///{os.path.abspath(os.path.join(cache_dir, cache_file))}"

    study_name = "correlation_threshold_optimization"

    # Load or create the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    logger.info(f"Study '{study_name}' loaded from {storage_url}.")

    # Calculate remaining trials
    remaining_trials = n_trials - len(study.trials)
    if remaining_trials <= 0:
        logger.info(
            f"Already completed {len(study.trials)} trials. No additional trials needed."
        )
    else:
        logger.info(f"Starting optimization with {remaining_trials} new trials...")
        # Create a copy of returns_df to prevent side effects
        returns_copy = copy.deepcopy(returns_df)

        # Define the objective function
        objective_func = functools.partial(
            objective,
            returns_df=returns_copy,
            performance_df=performance_df,
            market_returns=market_returns,
            risk_free_rate=risk_free_rate,
            sharpe_threshold=sharpe_threshold,
            linkage_method=linkage_method,
        )

        study.optimize(
            objective_func,
            n_trials=remaining_trials,
            n_jobs=1,  # Set to 1 to avoid SQLite locking issues
            timeout=None,  # Optional: set a timeout if needed
        )

    if study.best_trial:
        best_params = study.best_trial.params
        best_value = study.best_trial.value
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best objective value: {best_value}")
    else:
        logger.warning("No trials have been completed yet.")
        best_params = {}
        best_value = None

    return best_params, best_value


def objective(
    trial: optuna.trial.Trial,
    returns_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float,
    sharpe_threshold: float = 0.005,
    linkage_method: str = "average",
) -> float:
    """
    Objective function for Optuna to optimize correlation_threshold and lambda_weight.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        returns_df (pd.DataFrame): DataFrame containing returns of tickers.
        performance_df (pd.DataFrame): DataFrame containing performance metrics.
        market_returns (pd.Series): Series containing market returns.
        risk_free_rate (float): Risk-free rate for alpha calculation.
        sharpe_threshold (float, optional): Threshold for Sharpe Ratio in filtering. Defaults to 0.005.
        linkage_method (str, optional): Method for hierarchical clustering. Defaults to "average".

    Returns:
        float: Weighted objective to be maximized.
    """
    # Suggest correlation threshold and lambda weight
    correlation_threshold = trial.suggest_float("correlation_threshold", 0.5, 0.9)
    lambda_weight = trial.suggest_float("lambda", 0.1, 1.0)

    # Use a copy to prevent in-place modifications
    filtered_tickers = filter_correlated_groups(
        returns_df=returns_df.copy(),
        performance_df=performance_df.copy(),
        correlation_threshold=correlation_threshold,
        sharpe_threshold=sharpe_threshold,
        linkage_method=linkage_method,
    )

    # Prevent empty portfolio issue (return a small penalty instead of -inf)
    if not filtered_tickers or len(filtered_tickers) < 5:
        logger.warning(
            f"Too few tickers ({len(filtered_tickers)}) left after filtering. Applying penalty."
        )
        return -100  # Small penalty to discourage empty selections

    # Align stock histories before calculating correlation
    filtered_returns = returns_df[filtered_tickers].dropna(how="all")
    min_common_date = filtered_returns.dropna(how="all").index.min()
    filtered_returns = filtered_returns.loc[min_common_date:]

    # Ensure at least 50% of data is available per stock
    min_history = len(filtered_returns) * 0.5
    filtered_returns = filtered_returns.dropna(thresh=int(min_history), axis=1)

    if filtered_returns.empty:
        return -100  # Apply penalty if no stocks remain

    # Compute correlation on the largest available overlapping range
    avg_correlation = filtered_returns.corr().abs().mean().mean()

    # Calculate portfolio alpha
    portfolio_alpha = calculate_portfolio_alpha(
        filtered_returns, market_returns, risk_free_rate
    )

    # Weighted objective: maximize alpha and minimize correlation
    objective_value = portfolio_alpha - lambda_weight * avg_correlation

    logger.debug(
        f"Trial {trial.number}: correlation_threshold={correlation_threshold}, "
        f"lambda={lambda_weight}, portfolio_alpha={portfolio_alpha}, "
        f"avg_correlation={avg_correlation}, objective={objective_value}"
    )

    return objective_value
