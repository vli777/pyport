from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed
import json
from pathlib import Path

from anomaly.plot_anomalies import plot_anomalies
from anomaly.isolation_forest import apply_isolation_forest
from anomaly.plot_optimization_summary import plot_optimization_summary
from anomaly.anomaly_utils import detect_meme_stocks
from utils.performance_metrics import kappa_ratio
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


# Load the safe reference stocks
safe_reference_file_path = Path(__file__).parent / "safe_reference.json"
with safe_reference_file_path.open("r") as f:
    reference_stocks = json.load(f)
reference_stocks = [ticker.upper() for ticker in reference_stocks]


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    plot: bool = False,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/anomaly_thresholds.pkl",
    reoptimize: bool = False,  # Force re-optimization even if cache exists.
    global_filter: bool = False,  # If True, use one threshold for all stocks.
    max_anomaly_fraction: float = 0.03,  # Maximum allowed fraction of anomalies.
    contamination: Union[
        float, str, None
    ] = None,  # Contamination value for Isolation Forest.
) -> List[str]:
    """
    Filters out stocks with anomalous returns and returns a list of valid tickers.
    Uses per-ticker optimization to determine the threshold and caches all optimization
    results in a dictionary keyed by ticker. If a ticker’s anomaly fraction exceeds
    max_anomaly_fraction, it is removed.

    Args:
        returns_df (pd.DataFrame): Daily returns (columns are stocks).
        weight_dict (dict, optional): Objective function weights.
        plot (bool): If True, plot anomalies and optimization summary.
        n_jobs (int): Number of parallel jobs for per-ticker processing.
        cache_filename (str): File name for caching ticker info.
        reoptimize (bool): If True, force recalculation even if cached info exists.
        global_filter (bool): If True, optimize a single threshold using aggregated data.
        max_anomaly_fraction (float): Maximum allowed fraction of anomalies before dropping a ticker.
        contamination (float or str): Contamination parameter for Isolation Forest.

    Returns:
        List[str]: List of surviving (non-anomalous) ticker symbols.
    """
    if weight_dict is None:
        weight_dict = {"kappa": 0.8, "stability": 0.2}

    if contamination is None:
        meme_candidates = detect_meme_stocks(returns_df)
        # Adaptive contamination: Adjust based on % of flagged stocks
        contamination = min(
            0.02 + len(meme_candidates) / len(returns_df.columns) * 0.03, 0.05
        )
    elif contamination == "auto":
        # If explicitly set to "auto", allow it
        contamination = "auto"
    else:
        # Ensure it's a float within (0, 0.5]
        contamination = float(contamination)
        if not (0 < contamination <= 0.5):
            raise ValueError("contamination must be in the range (0, 0.5] or 'auto'.")

    # Load cached ticker info if available (cache is a dict keyed by ticker).
    cache: Dict[str, dict] = {}
    if not reoptimize:
        cache = load_parameters_from_pickle(cache_filename) or {}

    anomalous_cols = []  # Tickers flagged as anomalous

    # Global filtering: use a single threshold for all stocks.
    if global_filter:
        # Optimize a global threshold using all return data.
        combined_series = returns_df.stack().dropna()
        global_info = optimize_threshold_for_ticker(
            combined_series,
            weight_dict,
            stock="GLOBAL",
            reference_stocks=reference_stocks,
            contamination=contamination,
            max_anomaly_fraction=max_anomaly_fraction,
        )
        global_threshold = global_info["threshold"]

        # Apply the global threshold to each ticker.
        for stock in returns_df.columns:
            series = returns_df[stock].dropna()
            if series.empty:
                print(f"Warning: No data for stock {stock}. Skipping.")
                continue

            anomaly_flags, estimates = apply_isolation_forest(
                series,
                threshold=global_threshold,
                contamination=contamination,
            )
            ticker_info = {
                "threshold": global_threshold,
                "anomaly_flags": anomaly_flags,
                "estimates": estimates,
                "anomaly_fraction": anomaly_flags.mean(),
            }
            cache[stock] = ticker_info

            if ticker_info["anomaly_fraction"] > max_anomaly_fraction:
                anomalous_cols.append(stock)

    else:
        # Process each ticker individually.
        def process_ticker(stock: str):
            series = returns_df[stock].dropna()
            if series.empty:
                print(f"Warning: No data for stock {stock}. Skipping.")
                return None

            # Use cached info if available and not reoptimizing.
            if (stock in cache) and not reoptimize:
                ticker_info = cache[stock]
                thresh = ticker_info["threshold"]
            else:
                # Run optimization for this ticker.
                ticker_info = optimize_threshold_for_ticker(
                    series,
                    weight_dict,
                    stock,
                    reference_stocks=reference_stocks,
                    contamination=contamination,
                    max_anomaly_fraction=max_anomaly_fraction,
                )
                thresh = ticker_info["threshold"]

            anomaly_flags, estimates = apply_isolation_forest(
                series, threshold=thresh, contamination=contamination
            )
            ticker_info["anomaly_flags"] = anomaly_flags
            ticker_info["estimates"] = estimates
            ticker_info["anomaly_fraction"] = anomaly_flags.mean()

            return (stock, ticker_info)

        # Process tickers in parallel.
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(process_ticker)(stock) for stock in returns_df.columns
        )

        for res in parallel_results:
            if res is not None:
                stock, ticker_info = res
                cache[stock] = ticker_info
                if ticker_info["anomaly_fraction"] > max_anomaly_fraction:
                    anomalous_cols.append(stock)

    # Save the updated cache.
    save_parameters_to_pickle(cache, cache_filename)

    # Determine surviving tickers (those not flagged as anomalous).
    valid_tickers = [
        stock for stock in returns_df.columns if stock not in anomalous_cols
    ]

    if plot and cache:
        plot_anomalies(
            stocks=anomalous_cols,
            returns_data=returns_df.to_dict(orient="series"),
            anomaly_flags_data={
                stock: cache[stock]["anomaly_flags"] for stock in anomalous_cols
            },
            estimates_data={
                stock: cache[stock]["estimates"] for stock in anomalous_cols
            },
            thresholds={stock: cache[stock]["threshold"] for stock in cache},
        )
        # For the optimization summary, you could pass the list of all ticker info dicts.
        optimization_summary = list(cache.values())
        plot_optimization_summary(optimization_summary)

    return valid_tickers


def optimize_threshold_for_ticker(
    returns_series: pd.Series,
    weight_dict: Dict[str, float],
    stock: str,
    reference_stocks: List[str],
    contamination: float = "auto",
    max_anomaly_fraction: float = 0.02,
) -> float:
    """
    Optimizes the anomaly detection filter threshold for one ticker using Optuna.
    Uses the kappa ratio and applies a penalty that is a percentage
    of the composite score. For reference stocks, any anomaly flags reduce the composite score by 50%.
    For non-reference stocks, no anomalies trigger a 10% penalty.

    Args:
        returns_series (pd.Series): Series of returns.
        weight_dict (dict): Weights for objective components. Expected to contain keys "kappa" and "stability".
        stock (str): Stock symbol.
        reference_stocks (List[str]): List of reference stocks (e.g., safe stocks).
        contamination (float): Expected % of anomalies
        max_anomaly_fraction (float): Anomaly fraction threshold to be flagged

    Returns:
        dict: The optimal threshold and other study trial results
    """

    def objective(trial):
        # Suggest a threshold in a realistic range for Isolation Forest anomaly detection.
        threshold = trial.suggest_float("threshold", 5.0, 7.0, step=0.1)

        # Apply the Isolation Forest filter with the current threshold.
        anomaly_flags, scores = apply_isolation_forest(
            returns_series, threshold=threshold, contamination=contamination
        )
        mean_score = scores.mean()
        stdev_score = scores.std()
        anomaly_fraction = anomaly_flags.mean()

        # Save these values as user attributes so we can plot them later.
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", stdev_score)
        trial.set_user_attr("anomaly_fraction", anomaly_fraction)

        # Compute portfolio metrics using the kappa ratio.
        kappa = kappa_ratio(returns_series, order=3)
        rolling_volatility = returns_series.rolling(window=30).std().mean()
        stability_penalty = -rolling_volatility * 0.05

        # Compute composite performance without penalty.
        composite_without_penalty = (
            weight_dict["kappa"] * kappa + weight_dict["stability"] * stability_penalty
        )

        # Scale penalty up to a maximum if anomalies exceed a small fraction.
        if stock in reference_stocks:
            # For safe stocks, if anomaly_fraction is high, apply a penalty.
            penalty_factor = min(0.5, anomaly_fraction * 10.0)
        else:
            # For non-reference stocks, if anomaly_fraction is very low, apply a small penalty.
            penalty_factor = 0.1 if anomaly_fraction < max_anomaly_fraction else 0.0

        composite_score = composite_without_penalty * (1 - penalty_factor)

        trial.report(composite_score, step=int(threshold * 10))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return composite_score

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=50, timeout=60)

    if not study.best_trial:
        return {  # default
            "stock": stock,
            "threshold": 5.0,
            "best_score": 0.0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "anomaly_fraction": 0.0,
        }

    best_threshold = study.best_trial.params["threshold"]
    best_mean_score = study.best_trial.user_attrs.get("mean_score", 0.0)
    best_std_score = study.best_trial.user_attrs.get("std_score", 0.0)
    best_anomaly_fraction = study.best_trial.user_attrs.get("anomaly_fraction", 0.0)
    best_composite_score = study.best_value

    return {
        "stock": stock,
        "threshold": best_threshold,
        "best_score": best_composite_score,
        "mean_score": best_mean_score,
        "std_score": best_std_score,
        "anomaly_fraction": best_anomaly_fraction,
    }
