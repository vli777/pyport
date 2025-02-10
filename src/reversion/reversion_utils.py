from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

import hashlib
from datetime import datetime

from correlation.correlation_utils import compute_correlation_matrix
from utils.portfolio_utils import normalize_weights


def format_asset_cluster_map(
    asset_cluster_map: Dict[str, int], global_cache: Dict[str, Dict]
) -> Dict[str, List[str]]:
    """
    Formats the asset_cluster_map so that each noise ticker (-1) is treated as its own cluster.
    Also incorporates cluster information from global_cache to ensure consistency.

    Args:
        asset_cluster_map (dict): A dictionary mapping tickers to cluster IDs.
        global_cache (dict): Cached parameters, including cluster assignments.

    Returns:
        dict: A formatted dictionary where each noise ticker is its own cluster.
    """
    formatted_clusters = {}

    for ticker, cluster_id in asset_cluster_map.items():
        # If cluster info exists in global_cache, use it
        if ticker in global_cache and "cluster" in global_cache[ticker]:
            cluster_id = global_cache[ticker]["cluster"]

        # Ensure noise tickers (-1) are stored as their own clusters
        if cluster_id == -1 or cluster_id == np.int64(-1):
            formatted_clusters[ticker] = [ticker]
        else:
            formatted_clusters.setdefault(cluster_id, []).append(ticker)

    return formatted_clusters


def is_cache_stale(last_updated: str, max_age_days: int = 180) -> bool:
    """Check if the cache is stale based on the last update timestamp."""
    if not last_updated:  # Handle empty or None last_updated
        return True  # Treat missing timestamp as stale

    try:
        last_update = datetime.fromisoformat(last_updated)
    except ValueError:
        return True  # If it's an invalid timestamp, consider it stale

    return (datetime.now() - last_update).days >= max_age_days


def compute_ticker_hash(tickers: list) -> str:
    return hashlib.md5("".join(sorted(tickers)).encode("utf-8")).hexdigest()


def calculate_continuous_composite_signal(signals: dict, ticker_params: dict) -> dict:
    """
    Compute a composite mean reversion signal for each ticker.

    For each ticker, the composite signal is computed as:
         composite[ticker] = weight_daily * latest_daily_signal + weight_weekly * latest_weekly_signal

    Args:
        signals (dict): Mapping from ticker to its signals, e.g.
            {
                "AAPL": {"daily": {date: signal, ...}, "weekly": {date: signal, ...}},
                "MSFT": {"daily": {...}, "weekly": {...}},
                ...
            }
        ticker_params (dict): Global cache keyed by ticker with parameters, e.g.
            {
                "AAPL": {
                    "window_daily": 25,
                    "z_threshold_daily": 1.7,
                    "window_weekly": 25,
                    "z_threshold_weekly": 1.7,
                    "weight_daily": 0.5,
                    "weight_weekly": 0.5,
                },
                ...
            }
    Returns:
        dict: Mapping from ticker to its composite signal.
    """
    composite = {}
    for ticker, sig_data in signals.items():
        params = ticker_params.get(ticker, {})
        wd = params.get("weight_daily", 0.7)
        # Ensure weight_daily is within [0,1]
        wd = max(0.0, min(wd, 1.0))
        ww = params.get("weight_weekly", 0.3)

        # Retrieve the latest daily signal value, if available.
        daily_signal = sig_data.get("daily", {})
        daily_val = 0
        if daily_signal:
            # Assumes the keys are comparable dates (or timestamps)
            latest_date = max(daily_signal.keys())
            daily_val = daily_signal[latest_date]

        # Similarly for weekly signals.
        weekly_signal = sig_data.get("weekly", {})
        weekly_val = 0
        if weekly_signal:
            latest_date = max(weekly_signal.keys())
            weekly_val = weekly_signal[latest_date]

        composite[ticker] = wd * daily_val + ww * weekly_val
    return composite


def propagate_signals_by_similarity(
    composite_signals: dict,
    group_mapping: dict,
    baseline_allocation: Union[dict, pd.Series],
    returns_df: pd.DataFrame,
    signal_dampening: float = 0.5,
    lw_threshold: int = 50,
) -> dict:
    """
    Propagate composite signals within clusters by blending each ticker's own signal
    with a weighted average of the signals from the other tickers in the cluster. The
    weights come from the positive correlations (similarity) between tickers.

    This prevents a multiplicative effect when multiple source signals exist in the
    same cluster by normalizing the contribution.

    Args:
        composite_signals (dict): Original composite signals (ticker -> signal).
        group_mapping (dict): Mapping of cluster IDs to group info, which includes 'tickers'.
        baseline_allocation (dict or pd.Series): Baseline allocation (should be a Series).
        returns_df (pd.DataFrame): Returns DataFrame (dates as index, tickers as columns).
        lw_threshold (int): Size threshold for using Ledoit Wolf vs Pearson correlation.
        signal_dampening (float): Dampening factor when propagating signals

    Returns:
        dict: Updated composite signals with propagated, normalized values.
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

        # For each ticker, compute the weighted average of the signals from other tickers.
        for ticker in available_tickers:
            original_signal = composite_signals.get(ticker, 0)
            weighted_sum = 0.0
            sum_similarity = 0.0

            for source_ticker in available_tickers:
                if source_ticker == ticker:
                    continue

                source_signal = composite_signals.get(source_ticker, 0)
                if source_signal != 0:
                    # Get the similarity; default to 0 if not available.
                    similarity = 0
                    if (source_ticker in similarity_matrix.index) and (
                        ticker in similarity_matrix.columns
                    ):
                        similarity = similarity_matrix.at[source_ticker, ticker]
                    # Use only positive correlations.
                    if similarity > 0:
                        weighted_sum += source_signal * similarity
                        sum_similarity += similarity

            # Normalize the propagated signal by the total similarity weight.
            if sum_similarity > 0:
                propagated_signal = weighted_sum / sum_similarity
            else:
                propagated_signal = 0

            # Combine the original and the normalized propagated signal.
            updated_signals[ticker] = (
                1 - signal_dampening
            ) * original_signal + signal_dampening * propagated_signal

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

    normalized_adjusted = normalize_weights(adjusted)

    return normalized_adjusted


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
