from typing import Callable, Dict, Tuple
import optuna
import pandas as pd
import numpy as np
import functools


from correlation_utils import validate_matrix
from utils.portfolio_utils import calculate_portfolio_alpha
from utils import logger


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
    lambda_weight = trial.suggest_float("lambda", 0.1, 1.0)  # Range for lambda tuning
    
    # Apply the filter_correlated_groups function
    from correlation.decorrelation import filter_correlated_groups
    filtered_tickers = filter_correlated_groups(
        returns_df=returns_df.copy(),
        performance_df=performance_df.copy(),
        sharpe_threshold=sharpe_threshold,
        correlation_threshold=correlation_threshold,
        linkage_method=linkage_method,
        plot=False,
    )

    # If no tickers are left after filtering, return a minimal value
    if not filtered_tickers:
        logger.warning(
            f"No tickers left after filtering with correlation_threshold={correlation_threshold}. Returning -inf."
        )
        return float("-inf")

    # Calculate portfolio metrics
    filtered_returns = returns_df[filtered_tickers]
    avg_correlation = filtered_returns.corr().abs().mean().mean()
    portfolio_alpha = calculate_portfolio_alpha(
        filtered_returns, market_returns, risk_free_rate
    )

    # Weighted objective: maximize alpha and minimize correlation
    objective_value = portfolio_alpha - lambda_weight * avg_correlation

    logger.debug(
        f"Trial {trial.number}: correlation_threshold={correlation_threshold}, lambda={lambda_weight}, portfolio_alpha={portfolio_alpha}, avg_correlation={avg_correlation}, objective={objective_value}"
    )

    return objective_value


def optimize_correlation_threshold(
    returns_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float,
    sharpe_threshold: float = 0.005,
    linkage_method: str = "average",
    n_trials: int = 50,
    direction: str = "maximize",
    sampler: Callable = None,
    pruner: Callable = None,
    study_name: str = None,
    storage: str = None,
) -> Tuple[Dict[str, float], float]:
    """
    Optimize the correlation_threshold and lambda_weight to maximize the weighted objective using Optuna.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns of tickers.
        performance_df (pd.DataFrame): DataFrame containing performance metrics.
        market_returns (pd.Series): Series containing market returns.
        risk_free_rate (float): Risk-free rate for alpha calculation.
        sharpe_threshold (float, optional): Threshold for Sharpe Ratio in filtering. Defaults to 0.005.
        linkage_method (str, optional): Method for hierarchical clustering. Defaults to "average".
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        direction (str, optional): Optimization direction ("maximize" or "minimize"). Defaults to "maximize".
        sampler (Callable, optional): Optuna sampler. Defaults to None.
        pruner (Callable, optional): Optuna pruner. Defaults to None.
        study_name (str, optional): Name of the Optuna study for persistence. Defaults to None.
        storage (str, optional): Storage URL for Optuna study. Defaults to None.

    Returns:
        Tuple[Dict[str, float], float]: Best parameters and best objective value.
    """
    # Define a partial objective function with fixed parameters
    objective_func = functools.partial(
        objective,
        returns_df=returns_df,
        performance_df=performance_df,
        market_returns=market_returns,
        risk_free_rate=risk_free_rate,
        sharpe_threshold=sharpe_threshold,
        linkage_method=linkage_method,
    )

    # Create or load an Optuna study
    if storage and study_name:
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    # Optimize
    study.optimize(objective_func, n_trials=n_trials)

    # Log the best parameters and value
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best objective value: {study.best_value}")

    return study.best_params, study.best_value
