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
        group_signals (dict): Contains groups keyed by group label; each group is a dict with:
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
                    'group': 'Tech'
                },
                ...
            }
    Returns:
        dict: Mapping from ticker to composite continuous signal.
    """
    composite = {}
    # Iterate over groups in the group signals.
    for group_label, group_data in group_signals.items():
        daily_signals = group_data.get("daily", {})
        weekly_signals = group_data.get("weekly", {})
        for ticker in group_data.get("tickers", []):
            # Look up ticker-specific weights from the global cache.
            params = ticker_params.get(ticker, {})
            wd = params.get("weight_daily", 0.5)
            # Optionally ensure wd is in [0,1]
            wd = max(0.0, min(wd, 1.0))
            ww = params.get("weight_weekly", 0.5)
            # Use the latest available signal in each timeframe.
            daily_val = 0
            weekly_val = 0
            if ticker in daily_signals and daily_signals[ticker]:
                # Assuming the keys are dates and higher is later.
                latest_date = max(daily_signals[ticker])
                daily_val = daily_signals[ticker][latest_date]
            if ticker in weekly_signals and weekly_signals[ticker]:
                latest_date = max(weekly_signals[ticker])
                weekly_val = weekly_signals[ticker][latest_date]
            composite[ticker] = wd * daily_val + ww * weekly_val
    return composite


def adjust_allocation_with_mean_reversion(
    baseline_allocation: pd.Series,
    composite_signals: dict,
    alpha: float = 0.2,
    allow_short: bool = False,
) -> pd.Series:
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
