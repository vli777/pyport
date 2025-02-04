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
    group_signals: dict, group_weights: dict
) -> dict:
    """
    Compute a continuous composite mean reversion signal from the clustered signals.
    Instead of using discrete values, we expect the daily and weekly signals to be continuous
    (e.g. z-scores). The composite signal for each ticker is computed as:
        composite[ticker] = wd * daily_signal + (1 - wd) * weekly_signal
    where wd is the optimized weight for the daily signal.

    Args:
        group_signals (dict): For each group, contains:
            {
                "tickers": [list of tickers],
                "daily": {ticker: {date: continuous signal}, ...},
                "weekly": {ticker: {date: continuous signal}, ...},
            }
        group_weights (dict): Optimized daily/weekly weighting parameters per group.

    Returns:
        dict: Mapping from ticker to continuous composite signal.
    """
    composite = {}
    for group_label, weights in group_weights.items():
        if group_label not in group_signals:
            continue  # Skip if the group is missing.
        group_data = group_signals[group_label]
        daily = group_data.get("daily", {})
        weekly = group_data.get("weekly", {})

        wd = weights.get("weight_daily", 0.5)
        wd = max(0.0, min(wd, 1.0))  # Ensure within [0,1]
        ww = 1.0 - wd

        for ticker in group_data.get("tickers", []):
            # Use the latest available signal from each timeframe.
            daily_val = (
                daily[ticker][max(daily[ticker])]
                if ticker in daily and daily[ticker]
                else 0
            )
            weekly_val = (
                weekly[ticker][max(weekly[ticker])]
                if ticker in weekly and weekly[ticker]
                else 0
            )
            # Compute continuous composite signal.
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
    Negative weights can be clipped if shorts are not allowed, and the allocation is re-normalized.

    Args:
        baseline_allocation (pd.Series): Baseline allocation (ticker -> weight).
        composite_signals (dict): Ticker -> continuous signal (e.g. z-score).
        alpha (float): Sensitivity parameter for the adjustment.
        allow_short (bool): If False, negative adjusted weights are clipped to zero.

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
