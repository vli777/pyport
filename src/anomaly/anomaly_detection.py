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
from utils.logger import logger
from utils.performance_metrics import kappa_ratio
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    plot: bool = False,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/anomaly_thresholds.pkl",
    reoptimize: bool = False,  # Force re-optimization even if cache exists.
    global_filter: bool = False,  # If True, use one threshold for all stocks.
    max_anomaly_fraction: float = 0.01,  # Maximum allowed fraction of anomalies.
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
        contamination = min(0.01 + np.log1p(len(returns_df.columns)) * 0.002, 0.01)

    elif contamination == "auto":
        contamination = "auto"
    else:
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
            contamination=contamination,
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
                    contamination=contamination,
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
    if anomalous_cols:
        logger.info(
            f"Removed {len(anomalous_cols)} stocks due to high anomaly fraction: {sorted(anomalous_cols)}"
        )
    else:
        logger.info("No stocks were removed.")

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
    contamination: float = "auto",
) -> dict:
    """
    Optimizes the anomaly detection filter threshold for one ticker using Optuna.
    Uses the kappa ratio and applies a penalty that is a percentage of the composite score.
    For reference stocks, any anomaly flags reduce the composite score by 50%.
    For non-reference stocks, if the adjusted anomaly fraction is below the maximum, a 10% penalty is applied.
    Additionally, stocks with high volatility relative to a median benchmark receive an extra penalty,
    so that meme stocks (which tend to be very volatile) are not rewarded with high composite scores.

    Returns:
        dict: A dictionary with keys "stock", "threshold", "best_score", and "anomaly_fraction".
    """

    def objective(trial):
        threshold = trial.suggest_float("threshold", 4.2, 6.9, step=0.1)

        anomaly_flags, scores = apply_isolation_forest(
            returns_series, threshold=threshold, contamination=contamination
        )
        anomaly_fraction = anomaly_flags.mean()

        # Compute kappa; if NaN, set to 0.
        kappa = kappa_ratio(returns_series, order=3)
        if np.isnan(kappa):
            kappa = 0.0

        # Compute rolling volatility (30-day window with minimum 5 periods).
        rolling_std_series = returns_series.rolling(window=30, min_periods=5).std()
        # Use nanmean and nanmedian to be safe.
        rolling_volatility = np.nanmean(rolling_std_series)
        median_volatility = np.nanmedian(rolling_std_series)
        if np.isnan(rolling_volatility):
            rolling_volatility = 0.0
        if np.isnan(median_volatility) or median_volatility == 0:
            median_volatility = 1.0

        # Stability penalty: more volatility is worse.
        stability_penalty = -rolling_volatility * 0.05

        # Extreme volatility penalty: use 95th percentile of daily absolute pct changes.
        pct_changes = returns_series.pct_change().abs().dropna()
        if len(pct_changes) > 0:
            extreme_vol = np.percentile(pct_changes, 95)
            if np.isnan(extreme_vol):
                extreme_volatility_penalty = 0.0
            else:
                extreme_volatility_penalty = -extreme_vol * 0.5
        else:
            extreme_volatility_penalty = 0.0

        # Meme stock penalty: based on the 95th percentile of absolute rolling z-scores.
        rolling_mean = returns_series.rolling(window=30, min_periods=5).mean()
        rolling_std = returns_series.rolling(window=30, min_periods=5).std()
        z_score_series = (returns_series - rolling_mean) / (rolling_std + 1e-8)
        z_score_series = z_score_series.abs().fillna(0)
        if len(z_score_series) > 0:
            perc_z = np.percentile(z_score_series, 95)
            if np.isnan(perc_z):
                meme_stock_penalty = 0.0
            else:
                meme_stock_penalty = -perc_z * 1.5
        else:
            meme_stock_penalty = 0.0

        # Anomaly fraction penalty: directly penalize higher anomaly fractions.
        anomaly_fraction_penalty = anomaly_fraction * 10

        # Composite score as a sum of weighted components.
        composite_score = (
            weight_dict["kappa"] * kappa
            + weight_dict["stability"] * stability_penalty
            + extreme_volatility_penalty
            + meme_stock_penalty
            - anomaly_fraction_penalty
        )

        # Optional: Normalize by a volatility ratio.
        vol_penalty_ratio = (
            rolling_volatility / median_volatility if median_volatility > 0 else 1.0
        )
        composite_score /= vol_penalty_ratio + 1e-8

        # Debug print.
        print(
            f"[DEBUG] Stock: {stock}, Threshold: {threshold:.2f}, "
            f"Score: {composite_score:.4f}, Anomaly Fraction: {anomaly_fraction:.4f}"
        )

        trial.set_user_attr("anomaly_fraction", anomaly_fraction)
        trial.report(composite_score, step=int(threshold * 10))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return composite_score

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=50, timeout=60)

    if not study.best_trial or study.best_value == -np.inf:
        return {
            "stock": stock,
            "threshold": 5.0,
            "best_score": 0.0,
            "anomaly_fraction": 1.0,
        }

    best_threshold = study.best_trial.params["threshold"]
    best_anomaly_fraction = study.best_trial.user_attrs.get("anomaly_fraction", 0.0)
    best_composite_score = study.best_value

    return {
        "stock": stock,
        "threshold": best_threshold,
        "best_score": best_composite_score,
        "anomaly_fraction": best_anomaly_fraction,
    }
