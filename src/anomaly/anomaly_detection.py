from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed

from anomaly.plot_anomalies import plot_anomalies
from anomaly.isolation_forest import apply_isolation_forest
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle

FANGS = [
    "AAPL",
    "META",
    "AMZN",
    "GOOG",
    "GOOGL",
    "NFLX",
    "NVDA",
    "MSFT",
    "TSM",
    "BKNG",
    "TSLA",
]


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    plot: bool = False,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/anomaly_thresholds.pkl",
    reoptimize: bool = False,  # Force re-optimization even if cache exists
    global_filter: bool = False,  # If True, use a single threshold for all stocks
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Removes stocks with anomalous returns based on an anomaly detection filter.
    Optimizes thresholds per ticker using Optuna and caches the results.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns (columns = stocks).
        weight_dict (dict, optional): Objective function weights.
        plot (bool): If True, plots anomalies.
        n_jobs (int): Number of parallel jobs.
        cache_filename (str): Filename for threshold cache.
        reoptimize (bool): If True, force re-optimization.
        global_filter (bool): If True, use one threshold for all stocks.

    Returns:
        Tuple[pd.DataFrame, list, dict]: Filtered DataFrame, list of removed stocks, and thresholds.
    """
    if weight_dict is None:
        weight_dict = {"sortino": 0.8, "stability": 0.2}

    # Load cached thresholds if available.
    thresholds = {}
    if not reoptimize:
        thresholds = load_parameters_from_pickle(cache_filename) or {}

    anomalous_cols = []
    results = {}
    estimates_dict = {}

    if global_filter:
        # Use aggregated returns data to optimize a global threshold.
        combined_series = returns_df.stack().dropna()
        global_threshold = optimize_threshold_for_ticker(
            combined_series, weight_dict, stock="GLOBAL", reference_stocks=FANGS
        )
        # Save the same threshold for every stock.
        thresholds = {stock: global_threshold for stock in returns_df.columns}

        # Apply the global threshold to each ticker.
        for stock in returns_df.columns:
            series = returns_df[stock].dropna()
            if series.empty:
                print(f"Warning: No data for stock {stock}. Skipping.")
                continue
            anomaly_flags, estimates = apply_isolation_forest(
                series, threshold=global_threshold
            )
            if anomaly_flags.sum() > 0:
                anomalous_cols.append(stock)
                results[stock] = anomaly_flags
                estimates_dict[stock] = estimates
    else:
        # Define a helper function to process each ticker.
        def process_ticker(stock: str):
            series = returns_df[stock].dropna()
            if series.empty:
                print(f"Warning: No data for stock {stock}. Skipping.")
                return None

            # Use cached threshold if available.
            if (stock in thresholds) and not reoptimize:
                thresh = thresholds[stock]
            else:
                thresh = optimize_threshold_for_ticker(
                    series, weight_dict, stock, reference_stocks=FANGS
                )
            # Apply the filter.
            anomaly_flags, estimates = apply_isolation_forest(series, threshold=thresh)
            # Return the threshold even if no anomalies are flagged.
            return (stock, anomaly_flags, estimates, thresh)

        # Process each ticker in parallel.
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(process_ticker)(stock) for stock in returns_df.columns
        )

        # Collect results and update thresholds.
        for res in parallel_results:
            if res is not None:
                stock, anomaly_flags, estimates, thresh = res
                thresholds[stock] = thresh
                if anomaly_flags is not None and anomaly_flags.sum() > 0:
                    anomalous_cols.append(stock)
                    results[stock] = anomaly_flags
                    estimates_dict[stock] = estimates

    print(f"Removing {len(anomalous_cols)} stocks with anomalies: {anomalous_cols}")

    if plot and results:
        plot_anomalies(
            stocks=anomalous_cols,
            returns_data=returns_df.to_dict(orient="series"),
            anomaly_flags_data=results,
            estimates_data=estimates_dict,
            thresholds=thresholds,
            stocks_per_page=36,
        )

    # Cache the updated thresholds.
    save_parameters_to_pickle(thresholds, cache_filename)

    # Remove stocks flagged as anomalous.
    filtered_df = returns_df.drop(columns=anomalous_cols)
    return filtered_df, anomalous_cols, thresholds


def optimize_threshold_for_ticker(
    returns_series: pd.Series,
    weight_dict: Dict[str, float],
    stock: str,
    reference_stocks: List[str],
) -> float:
    """
    Optimizes the anomaly detection filter threshold for one ticker using Optuna.
    Ensures reference stocks (e.g. FANGS) are not flagged as anomalous.

    Args:
        returns_series (pd.Series): Series of returns.
        weight_dict (dict): Weights for objective components.
        stock (str): Stock symbol.
        reference_stocks (List[str]): List of reference stocks.

    Returns:
        float: The optimal threshold.
    """

    def objective(trial):
        # Choose a threshold in a realistic range.
        threshold = trial.suggest_float("threshold", 7.0, 12.0, step=0.1)
        anomaly_flags, _ = apply_isolation_forest(returns_series, threshold=threshold)
        num_anomalies = anomaly_flags.sum()

        # For reference stocks, heavily penalize if any anomaly is flagged.
        if stock in reference_stocks and num_anomalies > 0:
            return -np.inf
        # For non-reference stocks, penalize if no anomalies are found.
        if stock not in reference_stocks and num_anomalies == 0:
            return -np.inf

        # Compute portfolio metrics.
        portfolio_return = returns_series.mean()
        negative_returns = returns_series[returns_series < 0]
        downside_risk = negative_returns.std()
        sortino_ratio = portfolio_return / downside_risk if downside_risk != 0 else 0
        rolling_volatility = returns_series.rolling(window=30).std().mean()
        stability_penalty = -rolling_volatility * 0.05

        composite_score = (weight_dict["sortino"] * sortino_ratio) + (
            weight_dict["stability"] * stability_penalty
        )
        trial.report(composite_score, step=int(threshold * 10))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return composite_score

    sampler = TPESampler(seed=42)
    pruner = (
        NopPruner()
        if stock in reference_stocks
        else MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=50, timeout=60)
    return study.best_trial.params["threshold"] if study.best_trial else 10.0
