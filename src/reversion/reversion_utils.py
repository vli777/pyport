from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import hashlib

from correlation.correlation_utils import compute_correlation_matrix


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
            wd = params.get("weight_daily", 0.7)
            # Ensure weight_daily is within [0,1]
            wd = max(0.0, min(wd, 1.0))
            ww = params.get("weight_weekly", 0.3)
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


def propagate_signals_by_similarity(
    composite_signals: dict,
    group_mapping: dict,
    baseline_allocation: Union[dict, pd.Series],
    returns_df: pd.DataFrame,
    lw_threshold: int = 50,
) -> dict:
    """
    Propagate composite signals within clusters by adding each ticker's own signal
    with contributions from other tickers in the cluster weighted by their positive
    correlations. This ensures that if one ticker is strongly overbought (negative)
    or oversold (positive), highly correlated tickers will receive a similar signal.

    Args:
        composite_signals (dict): Original composite signals (ticker -> signal).
        group_mapping (dict): Mapping of cluster IDs to group info, which includes 'tickers'.
        baseline_allocation (dict or pd.Series): Baseline allocation (should be a Series).
        returns_df (pd.DataFrame): Returns DataFrame (dates as index, tickers as columns).
        lw_threshold (int): Size threshold for using Ledoit Wolf vs Pearson correlation.

    Returns:
        dict: Updated composite signals with propagated, correlation-weighted values.
    """
    updated_signals = composite_signals.copy()

    # Ensure baseline_allocation is a pandas Series, though it isn't used directly here.
    if isinstance(baseline_allocation, dict):
        baseline_allocation = pd.Series(baseline_allocation)

    for cluster_id, group_data in group_mapping.items():
        tickers_in_group = group_data.get("tickers", [])
        # Only keep tickers that exist in the returns data.
        available_tickers = [
            ticker for ticker in tickers_in_group if ticker in returns_df.columns
        ]
        if not available_tickers:
            continue

        # Subset returns for available tickers and drop columns that are completely NA.
        cluster_returns = returns_df[available_tickers].dropna(how="all", axis=1)
        if cluster_returns.empty:
            continue

        # Compute correlation (similarity) matrix for the cluster.
        similarity_matrix = compute_correlation_matrix(
            cluster_returns, lw_threshold=lw_threshold
        )

        # For each ticker, add contributions from every other ticker (with a nonzero signal)
        # weighted by the positive similarity.
        for ticker in available_tickers:
            propagated_signal = 0
            for source_ticker in available_tickers:
                if source_ticker == ticker:
                    continue
                source_signal = composite_signals.get(source_ticker, 0)
                if source_signal != 0:
                    # Get similarity value; if missing, default to 0.
                    similarity = 0
                    if (source_ticker in similarity_matrix.index) and (
                        ticker in similarity_matrix.columns
                    ):
                        similarity = similarity_matrix.at[source_ticker, ticker]
                    # Use only positive correlations so that the sign of the signal remains intact.
                    if similarity > 0:
                        propagated_signal += source_signal * similarity

            # The final updated signal is the ticker's own signal plus the propagated contributions.
            original_signal = composite_signals.get(ticker, 0)
            updated_signals[ticker] = original_signal + propagated_signal

    return updated_signals


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
    # Ensure composite_signals is a Pandas Series with tickers as index
    composite_signals = pd.Series(composite_signals)

    # Ensure baseline_allocation is also a Pandas Series
    if isinstance(baseline_allocation, dict):
        baseline_allocation = pd.Series(baseline_allocation)

    # Apply mean reversion adjustment
    adjusted = baseline_allocation.copy()
    for ticker in adjusted.index:
        signal = composite_signals.get(ticker, 0)  # Default to 0 if missing
        adjusted[ticker] *= 1 + alpha * signal

    if not allow_short:
        adjusted = adjusted.clip(lower=0)
        # Normalize so that the sum of weights equals 1.
        total = adjusted.sum()
        if total > 0:
            adjusted /= total
    else:
        # When allowing shorts, normalize by the sum of absolute weights
        total = adjusted.abs().sum()
        if total > 0:
            adjusted /= total

    return adjusted


def group_ticker_params_by_cluster(ticker_params: dict) -> dict:
    """
    Convert a global cache keyed by ticker into a dictionary keyed by cluster id.
    Each value is a dictionary with keys:
      - "tickers": a list of tickers in that cluster
      - "params": the parameters for that cluster (assumed to be the same for all tickers in the cluster)
    """
    group_parameters = {}
    for ticker, params in ticker_params.items():
        cluster = params.get("cluster", "Unknown")
        if cluster not in group_parameters:
            group_parameters[cluster] = {"tickers": [], "params": params}
        group_parameters[cluster]["tickers"].append(ticker)
    return group_parameters
