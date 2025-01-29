import os
from pathlib import Path
from typing import Dict, List, Tuple
import optuna
import pandas as pd

from utils import logger


def find_optimal_inclusion_pct(
    final_signals: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    storage_path: str = "optuna_cache/inclusion_thresholds.db",
    study_name: str = "inclusion_thresholds",
) -> Dict[str, float]:
    """
    Uses Optuna to optimize the inclusion/exclusion percentiles with persistent caching.

    Args:
        final_signals (pd.DataFrame): Weighted signal scores per ticker.
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of trials for Optuna optimization. Defaults to 50.
        storage_path (str): SQLite storage path for caching.
        study_name (str): Name of the Optuna study.

    Returns:
        Dict[str, float]: A dictionary containing the best inclusion/exclusion percentiles.
    """
    # Ensure cache directory exists
    cache_dir = Path(storage_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Define full storage path with SQLite URL format
    storage_url = f"sqlite:///{os.path.abspath(storage_path)}"

    # Load or create the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
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
        study.optimize(
            lambda trial: objective(trial, final_signals, returns_df),
            n_trials=remaining_trials,
            n_jobs=1,  # Set to 1 to avoid SQLite locking issues
            timeout=None,  # Optional: set a timeout if needed
        )

    # Retrieve the best parameters from the study
    if study.best_trial:
        best_include_pct = study.best_trial.params.get("include_threshold_pct", None)
        best_exclude_pct = study.best_trial.params.get("exclude_threshold_pct", None)

        optimal_thresholds = {
            "include_threshold_pct": best_include_pct,
            "exclude_threshold_pct": best_exclude_pct,
        }

        logger.info(f"Optimal thresholds found and saved: {optimal_thresholds}")
    else:
        logger.warning("No trials have been completed yet.")
        optimal_thresholds = {
            "include_threshold_pct": None,
            "exclude_threshold_pct": None,
        }

    return optimal_thresholds


def objective(trial, final_signals: pd.DataFrame, returns_df: pd.DataFrame) -> float:
    """
    Use Optuna to optimize the inclusion/exclusion thresholds while handling different stock history lengths.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        final_signals (pd.DataFrame): Weighted time-series signals per ticker.
        returns_df (pd.DataFrame): Log returns DataFrame.

    Returns:
        float: Cumulative return based on optimized thresholds.
    """
    # Ensure `final_signals` has a datetime index
    final_signals = final_signals.copy()
    final_signals.index = pd.to_datetime(final_signals.index)

    # Search space for inclusion/exclusion thresholds (percentiles)
    include_threshold_pct = trial.suggest_float(
        "include_threshold_pct", 0.1, 0.4, step=0.05
    )
    exclude_threshold_pct = trial.suggest_float(
        "exclude_threshold_pct", 0.1, 0.4, step=0.05
    )

    # Initialize positions DataFrame with NaN instead of zeros to allow dynamic updates
    positions = pd.DataFrame(
        index=returns_df.index, columns=returns_df.columns, dtype=float
    )

    for date in returns_df.index:
        date = pd.to_datetime(date)  # Ensure correct format

        # Get available stocks at this date (ignore missing values)
        available_stocks = returns_df.loc[date].dropna().index
        if date not in final_signals.index:
            continue  # Skip if no signal for this date

        current_signals = final_signals.loc[date, available_stocks].dropna()

        if current_signals.empty:
            continue  # No valid signals for this date

        include_threshold = current_signals.quantile(1 - include_threshold_pct)
        exclude_threshold = current_signals.quantile(exclude_threshold_pct)

        include_tickers = current_signals[
            current_signals >= include_threshold
        ].index.tolist()
        exclude_tickers = current_signals[
            current_signals <= exclude_threshold
        ].index.tolist()

        # Ensure no ticker is in both
        include_tickers = list(set(include_tickers) - set(exclude_tickers))
        exclude_tickers = list(set(exclude_tickers) - set(include_tickers))

        # Update only available stocks at this date
        positions.loc[date, include_tickers] = 1
        positions.loc[date, exclude_tickers] = -1

    # Fill missing values with 0 (stocks with no positions remain neutral)
    positions.fillna(0, inplace=True)

    # Simulate strategy
    _, cumulative_return = simulate_strategy(returns_df, positions)

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
