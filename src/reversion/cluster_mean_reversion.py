from typing import Dict
import numpy as np
import pandas as pd

from reversion.optimize_window_threshold import optimize_robust_mean_reversion
from reversion.reversion_utils import (
    format_asset_cluster_map,
)
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

    # Format clusters using the asset_cluster_map.
    clusters = format_asset_cluster_map(asset_cluster_map, global_cache)
    print(f"{len(clusters)} clusters found")

    overall_signals = {}  # This will be a flat dict keyed by ticker.

    # Build a reverse mapping: for each ticker already in the cache, map ticker -> its params.
    cached_ticker_params = {}
    for ticker, params in global_cache.items():
        if isinstance(params, dict) and "cluster" in params:
            cached_ticker_params[ticker] = params

    for cluster_label, tickers in clusters.items():
        # Filter tickers to those available in returns_df.
        tickers_in_returns = [t for t in tickers if t in returns_df.columns]
        if not tickers_in_returns:
            continue

        group_returns: pd.DataFrame = returns_df[tickers_in_returns].dropna(
            how="all", axis=1
        )
        if group_returns.empty:
            continue

        # Look for any cached parameters for tickers in this cluster.
        group_params = None
        for ticker in tickers_in_returns:
            if ticker in cached_ticker_params:
                group_params = cached_ticker_params[ticker]
                break

        if group_params is not None:
            # Propagate cached parameters to every ticker in this cluster.
            for ticker in tickers_in_returns:
                global_cache[ticker] = group_params

            group_signal = compute_group_stateful_signals(
                group_returns=group_returns,
                tickers=tickers_in_returns,
                params=group_params,
            )
        else:
            # Optimize the parameters for the group.
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

            # Build DataFrame for weight optimization.
            daily_signals_df = pd.DataFrame.from_dict(
                {
                    t: group_returns[t].to_dict()
                    for t in tickers_in_returns
                    if t in group_returns.columns
                },
                orient="index",
            ).T.fillna(0)
            # For weekly signals, for now, copy the daily signals.
            weekly_signals_df = daily_signals_df.copy()

            group_weights = find_optimal_weights(
                daily_signals_df=daily_signals_df,
                weekly_signals_df=weekly_signals_df,
                returns_df=group_returns,
                n_trials=n_trials,
                n_jobs=n_jobs,
            )

            # Combine optimized parameters into one dict for this group.
            # Notice that we now have separate thresholds for negative and positive signals.
            group_params = {
                "window_daily": round(best_params_daily.get("window", 20), 1),
                "z_threshold_daily_negative": round(
                    best_params_daily.get("z_threshold_negative", 1.5), 1
                ),
                "z_threshold_daily_positive": round(
                    best_params_daily.get("z_threshold_positive", 1.5), 1
                ),
                "window_weekly": round(best_params_weekly.get("window", 5), 1),
                "z_threshold_weekly_negative": round(
                    best_params_weekly.get("z_threshold_negative", 1.5), 1
                ),
                "z_threshold_weekly_positive": round(
                    best_params_weekly.get("z_threshold_positive", 1.5), 1
                ),
                "weight_daily": round(group_weights.get("weight_daily", 0.7), 1),
                "weight_weekly": round(group_weights.get("weight_weekly", 0.3), 1),
                "cluster": cluster_label,  # Only used during optimization.
            }

            # Update the cache for each ticker in this group.
            for ticker in tickers_in_returns:
                global_cache[ticker] = group_params

            group_signal = compute_group_stateful_signals(
                group_returns=group_returns,
                tickers=tickers_in_returns,
                params=group_params,
            )

        # Flatten the group_signal so that each ticker becomes its own key.
        # Expected format of group_signal:
        # {
        #   "tickers": [list of tickers],
        #   "daily": { ticker: daily_signal, ... },
        #   "weekly": { ticker: weekly_signal, ... }
        # }
        for ticker in group_signal.get("tickers", []):
            overall_signals[ticker] = {
                "daily": group_signal["daily"].get(ticker, {}),
                "weekly": group_signal["weekly"].get(ticker, {}),
            }
        print(f"Group {cluster_label}: {len(tickers_in_returns)} tickers optimized.")

    return overall_signals
