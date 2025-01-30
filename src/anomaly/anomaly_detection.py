import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from concurrent.futures import ThreadPoolExecutor

from anomaly.plot_anomalies import plot_anomalies
from anomaly.kalman_filter import apply_kalman_filter


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    threshold: float = None,
    plot: bool = False,
    cache_dir: str = "optuna_cache/kalman_thresholds",
    cache_file: str = "kalman_study.db",
    n_jobs: int = 1,  # Number of parallel jobs
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Removes stocks with anomalous returns based on the Kalman filter.
    Uses cached threshold if available, otherwise optimizes a new one.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns.
        weight_dict (dict, optional): Dictionary with optional objective weights.
        threshold (float, optional): Predefined Kalman filter threshold. If None, it will be optimized.
        plot (bool): If True, anomalies will be plotted.
        cache_dir (str): Cache directory for threshold storage.
        cache_file (str): Cache filename.
        n_jobs (int): Number of parallel jobs.

    Returns:
        Tuple[pd.DataFrame, list[str]]: Filtered DataFrame and list of removed symbols.
    """
    if weight_dict is None:
        weight_dict = {"sortino": 0.8, "stability": 0.2}

    # Use cached threshold if available
    if threshold is None:
        threshold = optimize_kalman_threshold(
            returns_df=returns_df,
            n_trials=50,
            weight_dict=weight_dict,
            cache_dir=cache_dir,
            cache_file=cache_file,
            n_jobs=n_jobs,
        )

    anomalous_cols = []
    returns_data = {}
    anomaly_flags_data = {}

    # Define a helper function for parallel processing
    def process_stock(stock):
        returns_series = returns_df[stock].dropna()
        if returns_series.empty:
            print(f"Warning: No data for stock {stock}. Skipping.")
            return None
        anomaly_flags = apply_kalman_filter(returns_series, threshold=threshold)
        if anomaly_flags.any():
            return stock, returns_series, anomaly_flags
        return None

    # Use ThreadPoolExecutor for I/O bound or ProcessPoolExecutor for CPU bound tasks
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(process_stock, returns_df.columns))

    for result in results:
        if result is not None:
            stock, returns_series, anomaly_flags = result
            anomalous_cols.append(stock)
            if plot:
                returns_data[stock] = returns_series
                anomaly_flags_data[stock] = anomaly_flags

    print(
        f"Removing {len(anomalous_cols)} stocks with Kalman anomalies: {anomalous_cols}"
    )

    if plot and returns_data:
        plot_anomalies(
            stocks=anomalous_cols,
            returns_data=returns_data,
            anomaly_flags_data=anomaly_flags_data,
            stocks_per_page=36,
        )

    filtered_df = returns_df.drop(columns=anomalous_cols)
    return filtered_df, anomalous_cols


def optimize_kalman_threshold(
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    weight_dict: Optional[Dict[str, float]] = None,
    cache_dir: str = "optuna_cache/kalman_thresholds",
    cache_file: str = "kalman_study.db",
    n_jobs: int = 1,  # Number of parallel jobs
) -> float:
    """
    Optimize the Kalman filter threshold using a multi-objective approach with parallel execution.
    Uses Optuna's SQLite cache to store results and avoid redundant optimizations.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int): Number of Optuna trials.
        weight_dict (dict, optional): Dictionary with weight settings
            (e.g., {'sortino': 0.8, 'stability': 0.2}).
        cache_dir (str): Directory for caching Optuna studies.
        cache_file (str): Cache database filename.
        n_jobs (int): Number of parallel jobs.

    Returns:
        float: Best threshold value.
    """
    if weight_dict is None:
        weight_dict = {"sortino": 0.8, "stability": 0.2}

    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # SQLite storage path
    storage_path = f"sqlite:///{os.path.join(cache_dir, cache_file)}"
    study_name = "kalman_threshold_optimization"

    # Define a sampler and pruner for better performance
    sampler = TPESampler(n_startup_trials=10, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # Load or create the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # Optimize with parallelization
    study.optimize(
        lambda trial: objective(trial, returns_df, weight_dict),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best_threshold = study.best_trial.params["threshold"]
    print(f"Best Kalman threshold found: {best_threshold} with weights {weight_dict}")

    return best_threshold


def objective(
    trial: optuna.Trial,
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
) -> float:
    """
    Optimize Kalman filter threshold using a combined objective function.

    Balances:
    - Sortino Ratio (risk-adjusted return)
    - Stability Penalty (rolling volatility to avoid meme stocks)

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        returns_df (pd.DataFrame): DataFrame of daily returns.
        weight_dict (dict): Dictionary with weight settings. Defaults to {'sortino': 0.8, 'stability': 0.2}.

    Returns:
        float: Composite score (higher is better).
    """
    if weight_dict is None:
        weight_dict = {"sortino": 0.8, "stability": 0.2}

    # Normalize weights
    total_weight = sum(weight_dict.values())
    weight_sortino = weight_dict.get("sortino", 0.8) / total_weight
    weight_stability = weight_dict.get("stability", 0.2) / total_weight

    # Let Optuna optimize threshold
    threshold = trial.suggest_float("threshold", 7.0, 12.0, step=0.5)

    # Now pass threshold into remove_anomalous_stocks
    # Disable plotting and parallel jobs within objective
    filtered_df, _ = remove_anomalous_stocks(
        returns_df=returns_df, threshold=threshold, plot=False, n_jobs=1
    )

    if filtered_df.empty:
        return -np.inf  # Penalize empty selections

    # Portfolio return
    portfolio_return = filtered_df.mean(axis=1).mean()

    # Calculate downside deviation (only negative returns)
    negative_returns = filtered_df.mean(axis=1)[filtered_df.mean(axis=1) < 0]
    downside_risk = negative_returns.std()

    # Avoid division by zero
    if downside_risk == 0:
        return -np.inf  # Penalize cases where no downside risk is captured

    # Sortino Ratio (risk-adjusted return)
    sortino_ratio = portfolio_return / downside_risk

    # Stability Penalty (rolling volatility)
    rolling_volatility = filtered_df.mean(axis=1).rolling(window=30).std().mean()

    # Avoid division by zero
    if rolling_volatility == 0:
        rolling_volatility = 1e-6  # Prevent divide by zero

    stability_penalty = (
        -rolling_volatility * 0.1
    )  # Reduce weight of excessive volatility

    # Weighted sum of Sortino Ratio and Stability Penalty
    composite_score = (weight_sortino * sortino_ratio) + (
        weight_stability * stability_penalty
    )

    # Report intermediate values for pruning
    trial.report(composite_score, step=threshold)

    # Prune if the trial is not promising
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return composite_score
