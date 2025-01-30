import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed

from anomaly.plot_anomalies import plot_anomalies
from anomaly.kalman_filter import apply_kalman_filter
from anomaly.caching import load_thresholds_from_pickle, save_thresholds_to_pickle


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    plot: bool = False,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/anomaly_thresholds.pkl",
    reoptimize: bool = False,  # Flag to force re-optimization
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Removes stocks with anomalous returns based on the Kalman filter.
    Optimizes thresholds per ticker using Optuna and caches the results.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns.
        weight_dict (dict, optional): Dictionary with objective weights.
        plot (bool): If True, anomalies will be plotted.
        n_jobs (int): Number of parallel jobs.
        cache_filename (str): Filename for the Pickle cache.
        reoptimize (bool): If True, force re-optimization even if cache exists.

    Returns:
        Tuple[pd.DataFrame, list[str], Dict[str, float]]: Filtered DataFrame, list of removed symbols, and thresholds.
    """
    if weight_dict is None:
        weight_dict = {"sortino": 0.8, "stability": 0.2}

    # Load cached thresholds
    thresholds = {}
    if not reoptimize:
        thresholds = load_thresholds_from_pickle(cache_filename)

    anomalous_cols = []
    results = {}
    estimates_dict = {}

    # Define a helper function for processing each ticker
    def process_ticker(stock, cached_thresholds):
        returns_series = returns_df[stock].dropna()
        if returns_series.empty:
            print(f"Warning: No data for stock {stock}. Skipping.")
            return None

        # Check if threshold is cached
        if stock in cached_thresholds and not reoptimize:
            threshold = cached_thresholds[stock]
        else:
            # Optimize threshold using Optuna
            threshold = optimize_threshold_for_ticker(returns_series, weight_dict)
            thresholds[stock] = threshold  # Update thresholds dict

        # Apply Kalman filter with optimized threshold
        anomaly_flags, estimates = apply_kalman_filter(
            returns_series, threshold=threshold
        )

        # Count anomalies
        num_anomalies = anomaly_flags.sum()

        if num_anomalies > 0:
            return (stock, anomaly_flags, estimates, threshold)
        return None

    # Parallel processing using Joblib
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(process_ticker)(stock, thresholds) for stock in returns_df.columns
    )

    # Collect results
    for res in parallel_results:
        if res is not None:
            stock, anomaly_flags, estimates, threshold = res
            anomalous_cols.append(stock)
            results[stock] = anomaly_flags
            estimates_dict[stock] = estimates

    print(
        f"Removing {len(anomalous_cols)} stocks with Kalman anomalies: {anomalous_cols}"
    )

    if plot and results:
        plot_anomalies(
            stocks=anomalous_cols,
            returns_data=returns_df.to_dict(orient="series"),
            anomaly_flags_data=results,
            estimates_data=estimates_dict,
            thresholds=thresholds,
            stocks_per_page=36,
        )

    # Save updated thresholds to cache
    save_thresholds_to_pickle(thresholds, cache_filename)

    # Filter out anomalous stocks
    filtered_df = returns_df.drop(columns=anomalous_cols)
    return filtered_df, anomalous_cols, thresholds


def optimize_threshold_for_ticker(
    returns_series: pd.Series, weight_dict: Dict[str, float]
) -> float:
    """
    Optimize the Kalman filter threshold for a single ticker using Optuna.

    Args:
        returns_series (pd.Series): Series of returns for the ticker.
        weight_dict (Dict[str, float]): Weights for the objective function.

    Returns:
        float: Optimal threshold.
    """

    def objective(trial):
        # Suggest a threshold within a realistic range based on prior knowledge
        threshold = trial.suggest_float("threshold", 9.0, 13.0, step=0.1)

        # Apply Kalman filter and detect anomalies
        anomaly_flags, estimates = apply_kalman_filter(
            returns_series, threshold=threshold
        )

        # Compute metrics
        num_anomalies = anomaly_flags.sum()
        if num_anomalies == 0:
            return -np.inf  # Penalize no anomalies detected

        # Portfolio return (example metric)
        portfolio_return = returns_series.mean()

        # Calculate downside deviation (only negative returns)
        negative_returns = returns_series[returns_series < 0]
        downside_risk = negative_returns.std()

        # Sortino Ratio (risk-adjusted return)
        sortino_ratio = portfolio_return / downside_risk if downside_risk != 0 else 0

        # Rolling volatility
        rolling_volatility = returns_series.rolling(window=30).std().mean()

        # Stability penalty
        stability_penalty = -rolling_volatility * 0.1  # Adjust multiplier as needed

        # Composite score
        composite_score = (weight_dict["sortino"] * sortino_ratio) + (
            weight_dict["stability"] * stability_penalty
        )

        # Report intermediate values for pruning
        trial.report(composite_score, step=int(threshold * 10))

        # Prune if necessary
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return composite_score

    # Create a study (in-memory for speed; adjust storage for persistence if needed)
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Optimize
    study.optimize(
        objective, n_trials=20, timeout=60
    )  # Adjust n_trials and timeout as needed

    if study.best_trial is None:
        print("No best trial found. Returning default threshold.")
        return 10.0  # Default threshold if optimization fails

    return study.best_trial.params["threshold"]
