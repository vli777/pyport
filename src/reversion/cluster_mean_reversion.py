from typing import Dict
import numpy as np
import pandas as pd

from reversion.optimize_window_threshold import optimize_robust_mean_reversion
from reversion.reversion_utils import format_asset_cluster_map, group_ticker_params_by_cluster
from reversion.optimize_period_weights import find_optimal_weights
from reversion.reversion_signals import compute_group_stateful_signals


def cluster_mean_reversion(
    asset_cluster_map: Dict[str, int],
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    global_cache: dict = None,  # Global cache passed in.
):
    if global_cache is None:
        global_cache = {}

    # Pull existing per-ticker params (if any)
    existing_params = global_cache.get("params", {})

    # Format clusters using asset_cluster_map and any existing cached params.
    clusters = format_asset_cluster_map(asset_cluster_map, existing_params)
    print(f"{len(clusters)} clusters found")

    # Build a reverse mapping from ticker to group parameters from the cache.
    cached_ticker_params = {}
    known_group_params = group_ticker_params_by_cluster(existing_params)
    for group_id, group_data in known_group_params.items():
        for ticker in group_data.get("tickers", []):
            cached_ticker_params[ticker] = group_data.get("params")

    overall_signals = {}

    for cluster_label, tickers in clusters.items():
        # Filter tickers to those available in returns_df.
        tickers_in_returns = [t for t in tickers if t in returns_df.columns]
        if not tickers_in_returns:
            continue

        # Use tickers_in_returns consistently when building group_returns.
        group_returns = returns_df[tickers_in_returns].dropna(how="all", axis=1)
        if group_returns.empty:
            continue

        # Check if any ticker in the current cluster has cached parameters.
        group_params = None
        for ticker in tickers_in_returns:
            if ticker in cached_ticker_params:
                group_params = cached_ticker_params[ticker]
                break

        if group_params:
            # Propagate the found group parameters to all tickers in the cluster.
            for ticker in tickers_in_returns:
                global_cache[ticker] = group_params
            overall_signals[cluster_label] = compute_group_stateful_signals(
                group_returns=group_returns, tickers=tickers_in_returns, params=group_params
            )
            continue  # Skip re-optimization

        # Otherwise, no cached parameters exist for any ticker in the group.
        # Optimize for daily returns.
        best_params_daily, _ = optimize_robust_mean_reversion(
            returns_df=group_returns,
            test_window_range=range(5, 31, 5),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
        # Optimize for weekly returns.
        weekly_returns = group_returns.resample("W").last()
        best_params_weekly, _ = optimize_robust_mean_reversion(
            returns_df=weekly_returns,
            test_window_range=range(1, 26),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        # Build a DataFrame of daily signals from group_returns.
        daily_signals_df = pd.DataFrame.from_dict(
            {t: group_returns[t].to_dict() for t in tickers_in_returns if t in group_returns.columns},
            orient="index",
        ).T.fillna(0)
        # For now, use a copy for weekly signals (or compute true weekly signals if available).
        weekly_signals_df = daily_signals_df.copy()

        group_weights = find_optimal_weights(
            daily_signals_df=daily_signals_df,
            weekly_signals_df=weekly_signals_df,
            returns_df=group_returns,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        # Combine optimized parameters into one dict for the group.
        group_params = {
            "window_daily": round(best_params_daily.get("window", 20), 1),
            "z_threshold_daily": round(best_params_daily.get("z_threshold", 1.5), 1),
            "window_weekly": round(best_params_weekly.get("window", 5), 1),
            "z_threshold_weekly": round(best_params_weekly.get("z_threshold", 1.5), 1),
            "weight_daily": round(group_weights.get("weight_daily", 0.7), 1),
            "weight_weekly": round(group_weights.get("weight_weekly", 0.3), 1),
            "cluster": cluster_label,  # Store the current cluster label.
        }

        # Update the global cache with the new group parameters for each ticker.
        for ticker in tickers_in_returns:
            global_cache[ticker] = group_params

        overall_signals[cluster_label] = compute_group_stateful_signals(
            group_returns=group_returns, tickers=tickers_in_returns, params=group_params
        )
        print(f"Group {cluster_label}: {len(tickers_in_returns)} tickers optimized.")

    return overall_signals
