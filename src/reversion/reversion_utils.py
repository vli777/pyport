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


def calculate_composite_signal(group_signals: dict, group_weights: dict) -> dict:
    """
    Compute a composite mean reversion signal from group clusters of tickers across daily and weekly time periods.

    Args:
        group_signals (dict): {
            "tickers": [list of tickers],
            "daily": {ticker: {date: signal}, ...},
            "weekly": {ticker: {date: signal}, ...},
        }
        group_weights (dict): Daily and weekly optimal weights per asset group for combining into a composite score.

    Returns:
        dict: Mapping from ticker to composite signal.
    """
    composite = {}

    for group_label, weights in group_weights.items():
        if group_label not in group_signals:
            continue  # Skip missing groups

        # Get group-specific signals:
        group_data = group_signals[group_label]
        daily = group_data.get("daily", {})
        weekly = group_data.get("weekly", {})

        wd = weights.get("weight_daily", 0.5)
        wd = max(0.0, min(wd, 1.0))  # Ensure weight is within [0, 1]
        ww = 1.0 - wd

        # For each ticker in this group, combine the latest signals.
        for ticker in group_data.get("tickers", []):
            daily_val = 0
            if ticker in daily and daily[ticker]:
                latest_daily = max(daily[ticker])
                daily_val = daily[ticker][latest_daily]

            weekly_val = 0
            if ticker in weekly and weekly[ticker]:
                latest_weekly = max(weekly[ticker])
                weekly_val = weekly[ticker][latest_weekly]

            composite[ticker] = wd * daily_val + ww * weekly_val

    return composite


def adjust_allocation_with_mean_reversion(
    baseline_allocation: pd.Series,
    composite_signals: Dict[str, float],
    alpha: float = 0.2,
    allow_short: bool = False,
) -> pd.Series:
    """
    Adjust the baseline allocation weights using the composite stat arb signal.
    The adjustment is multiplicative:
        new_weight = baseline_weight * (1 + alpha * composite_signal)
    Optionally, if shorts are not allowed, negative adjusted weights are clipped to zero.
    Finally, the weights are renormalized to sum to one.

    Args:
        baseline_allocation (pd.Series): Series with index = ticker and values = baseline weights.
        composite_signals (dict): Mapping from ticker to composite signal (float).
        alpha (float): Sensitivity factor controlling how strongly the signal adjusts the weight.
        allow_short (bool): Whether negative weights (shorts) are allowed. If False, negative weights are set to 0.

    Returns:
        pd.Series: The adjusted and normalized allocation weights.
    """
    adjusted = baseline_allocation.copy()
    for ticker in adjusted.index:
        signal = composite_signals.get(ticker, 0)
        # Multiply the baseline weight by a factor that increases with a positive signal.
        adjusted[ticker] = adjusted[ticker] * (1 + alpha * signal)

    # If shorts are not allowed, clip negative weights to zero.
    if not allow_short:
        adjusted = adjusted.clip(lower=0)

    # Renormalize so that the weights sum to 1.
    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total
    return adjusted
