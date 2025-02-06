from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed


from anomaly.plot_anomalies import plot_anomaly_overview
from anomaly.isolation_forest import apply_isolation_forest
from anomaly.plot_optimization_summary import plot_optimization_summary
from anomaly.kalman_filter import apply_kalman_filter
from anomaly.anomaly_utils import apply_fixed_zscore, get_cache_filename
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
    Kalman Filter (KF), or fixed Z-score. Uses per-ticker optimization and caches
    method-specific parameters.

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
    # Use "auto" if contamination is None to avoid sklearn errors.
    contamination_val = contamination if contamination is not None else "auto"

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

    def process_ticker(stock: str) -> Optional[Dict[str, Any]]:
        series = returns_df[stock].dropna()
        if series.empty:
            logger.warning(f"No data for stock {stock}. Skipping.")
            return None

        if use_isolation_forest:
            if stock in cache and not reoptimize:
                ticker_info = cache[stock]
            else:
                # Optimize threshold dynamically for the given method.
                ticker_info = optimize_threshold_for_ticker(
                    series, weight_dict, stock, method, contamination=contamination_val
                )
            thresh = ticker_info["threshold"]
            anomaly_flags, estimates = apply_isolation_forest(
                series, threshold=thresh, contamination=contamination_val
            )

        elif use_kalman_filter:
            thresh = 7.0
            anomaly_flags, estimates = apply_kalman_filter(series, threshold=thresh)

        elif use_fixed_zscore:
            thresh = 3.0
            anomaly_flags, estimates = apply_fixed_zscore(series, threshold=thresh)
        else:
            logger.error("No anomaly detection method selected.")
            return None

        ticker_info = {
            "stock": stock,
            "threshold": thresh,
            "anomaly_flags": anomaly_flags,
            "estimates": estimates,
            "anomaly_fraction": float(anomaly_flags.mean()),
        }
        return ticker_info

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
        plot_optimization_summary(list(cache.values()))
    if plot and anomalous_stocks:
        plot_anomaly_overview(anomalous_stocks, cache, returns_df)

    return valid_tickers


def optimize_threshold_for_ticker(
    returns_series: pd.Series,
    weight_dict: dict,
    stock: str,
    method: str,
    contamination: float = "auto",
) -> dict:
    """
    Optimizes the anomaly detection threshold for a single ticker using Optuna.
    The objective function scores the threshold based on multiple factors (kappa,
    volatility penalties, and anomaly fraction). The search space is adjusted
    dynamically depending on the method:
        - IF: Isolation Forest (default range: 4.2 to 6.9)
        - KF: Kalman Filter (default range: 5.0 to 8.0)
        - Z-score: Fixed Z-score (default range: 2.0 to 4.0)

    Args:
        returns_series (pd.Series): Series of stock returns.
        weight_dict (dict): Weights for scoring components (e.g. "kappa", "stability").
        stock (str): Ticker symbol.
        method (str): One of "IF", "KF", or "Z-score".
        contamination (float): Contamination parameter for Isolation Forest.

    Returns:
        dict: Contains "stock", "threshold", "best_score", and "anomaly_fraction".
    """

    def objective(trial):
        # Adjust the search space and anomaly detection function based on method.
        if method == "IF":
            threshold = trial.suggest_float("threshold", 4.2, 6.9, step=0.1)
            anomaly_flags, _ = apply_isolation_forest(
                returns_series, threshold=threshold, contamination=contamination
            )
        elif method == "KF":
            threshold = trial.suggest_float("threshold", 5.0, 8.0, step=0.1)
            anomaly_flags, _ = apply_kalman_filter(returns_series, threshold=threshold)
        elif method == "Z-score":
            threshold = trial.suggest_float("threshold", 2.0, 4.0, step=0.1)
            anomaly_flags, _ = apply_fixed_zscore(returns_series, threshold=threshold)
        else:
            raise ValueError(f"Unsupported method: {method}")

        anomaly_fraction = anomaly_flags.mean()

        # Compute kappa ratio; if NaN, default to 0.
        kappa = kappa_ratio(returns_series, order=3)
        if np.isnan(kappa):
            kappa = 0.0

        # Rolling volatility computation (30-day window, min 5 observations).
        rolling_std_series = returns_series.rolling(window=30, min_periods=5).std()
        rolling_volatility = np.nanmean(rolling_std_series)
        median_volatility = np.nanmedian(rolling_std_series)
        if np.isnan(rolling_volatility):
            rolling_volatility = 0.0
        if np.isnan(median_volatility) or median_volatility == 0:
            median_volatility = 1.0

        # Stability penalty: higher volatility reduces the score.
        stability_penalty = -rolling_volatility * 0.05

        # Extreme volatility penalty based on 95th percentile of daily pct changes.
        pct_changes = returns_series.pct_change().abs().dropna()
        if len(pct_changes) > 0:
            extreme_vol = np.percentile(pct_changes, 95)
            extreme_volatility_penalty = (
                -extreme_vol * 0.5 if not np.isnan(extreme_vol) else 0.0
            )
        else:
            extreme_volatility_penalty = 0.0

        # Meme stock penalty based on rolling z-scores.
        rolling_mean = returns_series.rolling(window=30, min_periods=5).mean()
        rolling_std = returns_series.rolling(window=30, min_periods=5).std()
        z_score_series = (
            ((returns_series - rolling_mean) / (rolling_std + 1e-8)).abs().fillna(0)
        )
        if len(z_score_series) > 0:
            perc_z = np.percentile(z_score_series, 95)
            meme_stock_penalty = -perc_z * 1.5 if not np.isnan(perc_z) else 0.0
        else:
            meme_stock_penalty = 0.0

        # Direct penalty for a higher anomaly fraction.
        anomaly_fraction_penalty = anomaly_fraction * 10

        # Composite score: higher is better.
        composite_score = (
            weight_dict["kappa"] * kappa
            + weight_dict["stability"] * stability_penalty
            + extreme_volatility_penalty
            + meme_stock_penalty
            - anomaly_fraction_penalty
        )

        # Normalize by volatility ratio.
        vol_penalty_ratio = (
            rolling_volatility / median_volatility if median_volatility > 0 else 1.0
        )
        composite_score /= vol_penalty_ratio + 1e-8

        print(
            f"[DEBUG] Stock: {stock}, Method: {method}, Threshold: {threshold:.2f}, "
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

    # Use a default threshold if optimization fails.
    if not study.best_trial or study.best_value == -np.inf:
        default_threshold = 5.0 if method == "IF" else (7.0 if method == "KF" else 3.0)
        return {
            "stock": stock,
            "threshold": default_threshold,
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
