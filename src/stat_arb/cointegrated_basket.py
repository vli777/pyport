from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.ensemble import IsolationForest
import optuna

from utils.z_scores import calculate_robust_zscores
from stat_arb.cointegration_utils import get_cointegration_vector


def johansen_cointegration(
    prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1
) -> np.ndarray:
    """
    Runs Johansen's cointegration test on a DataFrame of log prices and returns the first cointegrating vector.

    Args:
        prices (pd.DataFrame): DataFrame with dates as index and assets as columns (should be log-prices).
        det_order (int): Deterministic trend order.
        k_ar_diff (int): Number of lag differences.

    Returns:
        np.ndarray: The cointegrating vector (first eigenvector).
    """
    if prices.shape[1] < 2:
        raise ValueError("Johansen test requires at least two asset price series.")

    result = coint_johansen(prices, det_order, k_ar_diff)
    # Select the eigenvector corresponding to the largest eigenvalue
    cointegration_vector = result.evec[:, 0]
    return cointegration_vector


def compute_basket_spread(
    log_prices: pd.DataFrame, cointegration_vector: np.ndarray
) -> pd.Series:
    """
    Computes the basket spread as the weighted sum of log prices.

    Args:
        log_prices (pd.DataFrame): Log-prices with dates as index and assets as columns.
        cointegration_vector (np.ndarray): Cointegrating vector.

    Returns:
        pd.Series: The basket spread time series.
    """
    spread = log_prices.dot(cointegration_vector)
    return spread


def detect_outlier(series: pd.Series, contamination: float = 0.05) -> int:
    """
    Uses an Isolation Forest to detect if the latest point of the series is an outlier.

    Args:
        series (pd.Series): The time series (e.g., basket spread).
        contamination (float): The proportion of outliers expected.

    Returns:
        int: The prediction for the latest data point (1 = inlier, -1 = outlier).
    """
    cleaned_series = series.dropna()
    if len(cleaned_series) < 2:
        return 1  # Not enough data to determine outliers, assume inlier

    values = cleaned_series.values.reshape(-1, 1)
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(values)

    latest_value = np.array([[cleaned_series.iloc[-1]]])
    flag = iso.predict(latest_value)[0]
    return flag


def analyze_basket(
    returns_df: pd.DataFrame,
    cache_path: Path,
    window: int = 30,
    zscore_threshold: float = 2.0,
    iso_contamination: float = 0.05,
    reoptimize: bool = False,
) -> dict:
    if returns_df.shape[1] < 2:
        raise ValueError("Cointegration analysis requires at least two stocks.")

    asset_names = returns_df.columns.tolist()
    log_prices = returns_df.cumsum()

    # Use the caching mechanism here
    coint_vector = get_cointegration_vector(returns_df, cache_path, reoptimize)

    spread = compute_basket_spread(log_prices, coint_vector)
    spread_z = calculate_robust_zscores(spread, window)
    iso_flag = detect_outlier(spread, contamination=iso_contamination)

    latest_z = spread_z.dropna().iloc[-1]
    latest_spread = spread.dropna().iloc[-1]

    if latest_z > zscore_threshold and iso_flag == -1:
        signal = "Short Basket (spread overvalued)"
    elif latest_z < -zscore_threshold and iso_flag == -1:
        signal = "Long Basket (spread undervalued)"
    else:
        signal = "No clear trade signal"

    hedge_ratios = coint_vector / np.abs(coint_vector).sum()
    hedge_ratios_dict = dict(zip(asset_names, hedge_ratios))
    cointegration_vector_dict = dict(zip(asset_names, coint_vector))

    return {
        "cointegration_vector": cointegration_vector_dict,
        "hedge_ratios": hedge_ratios_dict,
        "latest_spread": latest_spread,
        "latest_zscore": latest_z,
        "isolation_flag": iso_flag,
        "signal": signal,
        "spread_series": spread,
        "spread_zscore_series": spread_z,
    }


def objective(trial: optuna.Trial, prices_df: pd.DataFrame) -> float:
    """
    An example objective function to tune the rolling window, z-score threshold,
    and isolation forest contamination for the basket analysis.

    In practice, you would link the signal to a backtested performance metric.
    For demonstration purposes, we return the negative absolute latest z-score (i.e., favoring higher mean reversion).

    Args:
        trial (optuna.Trial): Optuna trial object.
        prices_df (pd.DataFrame): Price data for the basket.

    Returns:
        float: Objective value to minimize.
    """
    window = trial.suggest_int("window", 20, 60)
    zscore_threshold = trial.suggest_float("zscore_threshold", 1.5, 3.0)
    iso_contamination = trial.suggest_float("iso_contamination", 0.01, 0.1)

    result = analyze_basket(
        prices_df,
        window=window,
        zscore_threshold=zscore_threshold,
        iso_contamination=iso_contamination,
    )

    objective_value = -abs(result["latest_zscore"])
    return objective_value


def run_optuna_study_for_basket(prices_df: pd.DataFrame, n_trials: int = 50) -> dict:
    """
    Encapsulates running an Optuna study to tune basket analysis parameters.

    This function is intended to be called from your pipeline. It returns the best parameters and objective value.

    Args:
        prices_df (pd.DataFrame): Price data for the basket.
        n_trials (int): Number of Optuna trials.

    Returns:
        dict: Dictionary with keys "best_params" and "best_value".
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, prices_df), n_trials=n_trials, n_jobs=-1)

    return {"best_params": study.best_params, "best_value": study.best_value}
