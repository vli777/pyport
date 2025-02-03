from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed
import json
from pathlib import Path

from anomaly.plot_anomalies import plot_anomaly_overview
from anomaly.isolation_forest import apply_isolation_forest
from anomaly.plot_optimization_summary import plot_optimization_summary
from anomaly.kalman_filter import apply_kalman_filter
from src.anomaly.anomaly_utils import get_cache_filename
from utils.logger import logger
from utils.performance_metrics import kappa_ratio
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    plot: bool = False,
    n_jobs: int = -1,
    reoptimize: bool = False,
    max_anomaly_fraction: float = 0.01,
    contamination: Union[float, str, None] = None,
) -> List[str]:
    """
    Filters out stocks with anomalous returns using Isolation Forest (IF),
    Kalman Filter (KF), or Z-score, based on the number of stocks in the dataset.
    Uses per-ticker optimization and caches method-specific parameters.

    Args:
        returns_df (pd.DataFrame): DataFrame where each column corresponds to a stock's returns.
        weight_dict (Optional[Dict[str, float]]): Weights for optimization criteria.
        plot (bool): If True, generate anomaly and optimization plots.
        n_jobs (int): Number of parallel jobs to run.
        reoptimize (bool): If True, bypass cache and reoptimize thresholds.
        max_anomaly_fraction (float): Maximum fraction of anomalous data allowed per stock.
        contamination (Union[float, str, None]): Contamination parameter for Isolation Forest.

    Returns:
        List[str]: List of tickers that are not flagged as anomalous.
    """
    weight_dict = weight_dict or {"kappa": 0.8, "stability": 0.2}
    n_stocks = len(returns_df.columns)

    # Dynamically choose the anomaly detection method.
    if n_stocks > 20:
        method = "IF"
        use_isolation_forest, use_kalman_filter, use_fixed_zscore = True, False, False
    elif 5 < n_stocks <= 20:
        method = "KF"
        use_kalman_filter, use_isolation_forest, use_fixed_zscore = True, False, False
    else:
        method = "Z-score"
        use_fixed_zscore, use_isolation_forest, use_kalman_filter = True, False, False

    cache_filename = get_cache_filename(method)
    cache: Dict[str, Any] = (
        {} if reoptimize else load_parameters_from_pickle(cache_filename) or {}
    )

    # Process each ticker individually.
    def process_ticker(stock: str) -> Optional[Dict[str, Any]]:
        series = returns_df[stock].dropna()
        if series.empty:
            logger.warning(f"No data for stock {stock}. Skipping.")
            return None

        if use_isolation_forest:
            # Optimize threshold if not cached or if reoptimization is requested.
            if stock in cache and not reoptimize:
                ticker_info = cache[stock]
            else:
                ticker_info = optimize_threshold_for_ticker(
                    series, weight_dict, stock, contamination=contamination
                )
            thresh = ticker_info["threshold"]
            anomaly_flags, estimates = apply_isolation_forest(
                series, threshold=thresh, contamination=contamination
            )

        elif use_kalman_filter:
            thresh = 7.0  # Empirically determined threshold.
            anomaly_flags, estimates = apply_kalman_filter(series, threshold=thresh)

        else:  # Z-score method for very small datasets.
            thresh = 3.0
            residuals = (series - series.mean()) / series.std()
            anomaly_flags = (np.abs(residuals) > thresh).astype(bool)
            estimates = series.copy()

        ticker_info = {
            "stock": stock,
            "threshold": thresh,
            "anomaly_flags": anomaly_flags,
            "estimates": estimates,
            "anomaly_fraction": float(anomaly_flags.mean()),
        }
        return ticker_info

    # Parallel processing of each stock.
    processed_info = Parallel(n_jobs=n_jobs)(
        delayed(process_ticker)(stock) for stock in returns_df.columns
    )

    anomalous_stocks: List[str] = []
    for info in processed_info:
        if info is None:
            continue
        cache[info["stock"]] = info
        if info["anomaly_fraction"] > max_anomaly_fraction:
            anomalous_stocks.append(info["stock"])

    save_parameters_to_pickle(cache, cache_filename)
    valid_tickers = [
        stock for stock in returns_df.columns if stock not in anomalous_stocks
    ]
    logger.info(
        f"Removed {len(anomalous_stocks)} stocks due to high anomaly fraction: {sorted(anomalous_stocks)}"
        if anomalous_stocks
        else "No stocks were removed."
    )

    if plot and cache:
        # Plot anomalies using method-specific plotting function.
        plot_anomaly_overview(cache, returns_df)
        plot_optimization_summary(list(cache.values()))

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
