from typing import Dict
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def compute_distance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = returns_df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 1.0)
    distance_matrix = 1 - corr_matrix
    return distance_matrix


def cluster_stocks(
    returns_df: pd.DataFrame, eps: float = 0.2, min_samples: int = 2
) -> dict:
    distance_matrix = compute_distance_matrix(returns_df)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(distance_matrix)

    clusters = {}
    for ticker, label in zip(returns_df.columns, cluster_labels):
        clusters.setdefault(label, []).append(ticker)
    return clusters


def compute_ticker_hash(tickers: list) -> str:
    import hashlib

    return hashlib.md5("".join(sorted(tickers)).encode("utf-8")).hexdigest()


def calculate_continuous_composite_signal(
    group_signals: dict, ticker_params: dict
) -> dict:
    """
    Compute a continuous composite mean reversion signal for each ticker using
    per-ticker parameters from the global cache.

    For each ticker, the composite signal is computed as:
         composite[ticker] = weight_daily * daily_signal + weight_weekly * weekly_signal

    Args:
        group_signals (dict): Dictionary keyed by group label with values:
            {
                "tickers": [list of tickers],
                "daily": {ticker: {date: continuous signal}, ...},
                "weekly": {ticker: {date: continuous signal}, ...},
            }
        ticker_params (dict): Global cache keyed by ticker with parameters, e.g.
            {
                'AAPL': {
                    'window_daily': 25,
                    'z_threshold_daily': 1.7,
                    'window_weekly': 25,
                    'z_threshold_weekly': 1.7,
                    'weight_daily': 0.5,
                    'weight_weekly': 0.5,
                },
                ...
            }
    Returns:
        dict: Mapping from ticker to its composite signal.
    """
    composite = {}
    for group_label, group_data in group_signals.items():
        daily_signals = group_data.get("daily", {})
        weekly_signals = group_data.get("weekly", {})
        for ticker in group_data.get("tickers", []):
            params = ticker_params.get(ticker, {})
            wd = params.get("weight_daily", 0.5)
            # Ensure weight_daily is within [0,1]
            wd = max(0.0, min(wd, 1.0))
            ww = params.get("weight_weekly", 0.5)
            # If both weights are from the same group they should sum to ~1.
            # (They were set as weight_weekly = 1 - weight_daily in the optimization.)

            # Get the latest daily signal if available.
            daily_val = 0
            if ticker in daily_signals and daily_signals[ticker]:
                # Assuming dates are comparable (e.g., as strings or timestamps)
                latest_date = max(daily_signals[ticker].keys())
                daily_val = daily_signals[ticker][latest_date]
            # Similarly for weekly signals.
            weekly_val = 0
            if ticker in weekly_signals and weekly_signals[ticker]:
                latest_date = max(weekly_signals[ticker].keys())
                weekly_val = weekly_signals[ticker][latest_date]

            composite[ticker] = wd * daily_val + ww * weekly_val
    return composite


def adjust_allocation_with_mean_reversion(
    baseline_allocation: pd.Series,
    composite_signals: dict,
    alpha: float = 0.2,
    allow_short: bool = False,
) -> pd.Series:
    """
    Adjust the baseline allocation using a continuous mean reversion signal.
    The adjustment is multiplicative:
         new_weight = baseline_weight * (1 + alpha * composite_signal)
    Negative weights are clipped if shorts are not allowed, and the result is renormalized.

    Args:
        baseline_allocation (pd.Series): Series with index = ticker and values = baseline weights.
        composite_signals (dict): Mapping from ticker to continuous signal (e.g. a z-score).
        alpha (float): Sensitivity factor.
        allow_short (bool): If False, negative adjusted weights are set to zero.

    Returns:
        pd.Series: Adjusted and normalized allocation.
    """
    adjusted = baseline_allocation.copy()
    for ticker in adjusted.index:
        signal = composite_signals.get(ticker, 0)
        adjusted[ticker] = adjusted[ticker] * (1 + alpha * signal)

    if not allow_short:
        adjusted = adjusted.clip(lower=0)

    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total
    return adjusted
