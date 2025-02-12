from typing import Dict
import numpy as np
import pandas as pd

from reversion.optimize_window_threshold import optimize_robust_mean_reversion
from reversion.reversion_utils import (
    format_asset_cluster_map,
)
from reversion.reversion_signals import (
    compute_stateful_signal_with_decay,
)
from reversion.optimize_period_weights import find_optimal_weights


def cluster_mean_reversion(
    asset_cluster_map: Dict[str, int],
    returns_df: pd.DataFrame,
    objective_weights: dict,
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

    # Build a reverse mapping for cached parameters.
    cached_ticker_params = {
        ticker: params
        for ticker, params in global_cache.items()
        if isinstance(params, dict)
    }

    for cluster_label, tickers in clusters.items():
        # Filter tickers to those present in returns_df.
        tickers_in_returns = [t for t in tickers if t in returns_df.columns]
        if not tickers_in_returns:
            continue

        group_returns = returns_df[tickers_in_returns].dropna(how="all", axis=1)
        if group_returns.empty:
            continue

        # Try to retrieve cached parameters from any ticker in this cluster.
        group_params = None
        for ticker in tickers_in_returns:
            if ticker in cached_ticker_params:
                group_params = cached_ticker_params[ticker]
                break

        if group_params is not None:
            # Use cached parameters (which include the optimal weights).
            daily_params = {
                "window": group_params["window_daily"],
                "z_threshold_positive": group_params["z_threshold_daily_positive"],
                "z_threshold_negative": group_params["z_threshold_daily_negative"],
            }
            weekly_params = {
                "window": group_params["window_weekly"],
                "z_threshold_positive": group_params["z_threshold_weekly_positive"],
                "z_threshold_negative": group_params["z_threshold_weekly_negative"],
            }
            weight_daily = group_params["weight_daily"]
            weight_weekly = group_params["weight_weekly"]

        else:
            # Optimize daily parameters.
            best_params_daily, _ = optimize_robust_mean_reversion(
                returns_df=group_returns,
                objective_weights=objective_weights,
                test_window_range=range(5, 31, 5),
                n_trials=n_trials,
                n_jobs=n_jobs,
            )
            # Optimize weekly parameters on resampled data.
            weekly_returns = group_returns.resample("W").last()
            best_params_weekly, _ = optimize_robust_mean_reversion(
                returns_df=weekly_returns,
                objective_weights=objective_weights,
                test_window_range=range(1, 26),
                n_trials=n_trials,
                n_jobs=n_jobs,
            )

            daily_params = {
                "window": round(best_params_daily.get("window", 20), 1),
                "z_threshold_positive": round(
                    best_params_daily.get("z_threshold_positive", 1.5), 1
                ),
                "z_threshold_negative": round(
                    best_params_daily.get("z_threshold_negative", 1.5), 1
                ),
            }
            weekly_params = {
                "window": round(best_params_weekly.get("window", 5), 1),
                "z_threshold_positive": round(
                    best_params_weekly.get("z_threshold_positive", 1.5), 1
                ),
                "z_threshold_negative": round(
                    best_params_weekly.get("z_threshold_negative", 1.5), 1
                ),
            }

            # For weight optimization, compute temporary signals.
            daily_signals_opt = {}
            for t in tickers_in_returns:
                series = group_returns[t].dropna()
                daily_signals_opt[t] = (
                    compute_stateful_signal_with_decay(series, daily_params)
                    if not series.empty
                    else pd.Series(dtype=float)
                )
            daily_signals_df_opt = pd.concat(daily_signals_opt, axis=1).fillna(0)

            weekly_signals_opt = {}
            for t in tickers_in_returns:
                series_weekly = group_returns[t].resample("W").last().dropna()
                weekly_signals_opt[t] = (
                    compute_stateful_signal_with_decay(series_weekly, weekly_params)
                    if not series_weekly.empty
                    else pd.Series(dtype=float)
                )
            weekly_signals_df_opt = pd.concat(weekly_signals_opt, axis=1).fillna(0)

            best_period_weights = find_optimal_weights(
                daily_signals_df_opt,
                weekly_signals_df_opt,
                returns_df,
                objective_weights,
                n_trials=n_trials,
                n_jobs=n_jobs,
                group_id=cluster_label,
            )
            weight_daily = best_period_weights.get("weight_daily", 0.7)
            weight_weekly = 1.0 - weight_daily
            print(
                f"Optimized weights for cluster {cluster_label}: daily={weight_daily}, weekly={weight_weekly}"
            )

            # Store the newly optimized parameters.
            group_params = {
                "window_daily": daily_params["window"],
                "z_threshold_daily_positive": daily_params["z_threshold_positive"],
                "z_threshold_daily_negative": daily_params["z_threshold_negative"],
                "window_weekly": weekly_params["window"],
                "z_threshold_weekly_positive": weekly_params["z_threshold_positive"],
                "z_threshold_weekly_negative": weekly_params["z_threshold_negative"],
                "weight_daily": weight_daily,
                "weight_weekly": weight_weekly,
            }

        # Now compute the stateful signals (daily and weekly) using the chosen parameters.
        daily_signals = {}
        for t in tickers_in_returns:
            series = group_returns[t].dropna()
            daily_signals[t] = (
                compute_stateful_signal_with_decay(series, daily_params)
                if not series.empty
                else pd.Series(dtype=float)
            )
        weekly_signals = {}
        for t in tickers_in_returns:
            series_weekly = group_returns[t].resample("W").last().dropna()
            weekly_signals[t] = (
                compute_stateful_signal_with_decay(series_weekly, weekly_params)
                if not series_weekly.empty
                else pd.Series(dtype=float)
            )

        # Update the cache for each ticker in this cluster.
        for ticker in tickers_in_returns:
            global_cache[ticker] = group_params

        # Flatten the signals dictionary so each ticker has its own entry.
        for ticker in tickers_in_returns:
            overall_signals[ticker] = {
                "daily": daily_signals.get(ticker, {}),
                "weekly": weekly_signals.get(ticker, {}),
            }

    return overall_signals
