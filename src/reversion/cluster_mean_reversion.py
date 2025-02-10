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

        # Check for cached parameters in this cluster.
        group_params = None
        for ticker in tickers_in_returns:
            if ticker in cached_ticker_params:
                group_params = cached_ticker_params[ticker]
                break

        if group_params is not None:
            # Use the cached window/threshold parameters, including weights.
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
            print(
                f"DEBUG: Using cached window/threshold parameters for cluster {cluster_label}"
            )
        else:
            # Optimize window and thresholds.
            best_params_daily, _ = optimize_robust_mean_reversion(
                returns_df=group_returns,
                test_window_range=range(5, 31, 5),
                n_trials=n_trials,
                n_jobs=n_jobs,
            )
            weekly_returns = group_returns.resample("W").last()
            best_params_weekly, _ = optimize_robust_mean_reversion(
                returns_df=weekly_returns,
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
            # Set default weights (e.g. 0.7 for daily, 0.3 for weekly).
            group_params = {
                "window_daily": daily_params["window"],
                "z_threshold_daily_positive": daily_params["z_threshold_positive"],
                "z_threshold_daily_negative": daily_params["z_threshold_negative"],
                "window_weekly": weekly_params["window"],
                "z_threshold_weekly_positive": weekly_params["z_threshold_positive"],
                "z_threshold_weekly_negative": weekly_params["z_threshold_negative"],
                "weight_daily": 0.7,  # default weight (can be tuned offline)
                "weight_weekly": 0.3,
            }
            print(
                f"DEBUG: Optimized window/threshold parameters for cluster {cluster_label}"
            )

        # --- Compute stateful signals using the (cached or optimized) parameters ---

        # Compute daily signals.
        daily_signals = {}
        for t in tickers_in_returns:
            series = group_returns[t].dropna()
            if series.empty:
                daily_signals[t] = pd.Series(dtype=float)
            else:
                daily_signals[t] = compute_stateful_signal_with_decay(
                    series, daily_params
                )
        daily_signals_df = pd.concat(daily_signals, axis=1).fillna(0)
        print(
            f"Daily signals summary for cluster {cluster_label}:\n",
            daily_signals_df.describe(),
        )

        # Compute weekly signals.
        weekly_signals = {}
        for t in tickers_in_returns:
            series_weekly = group_returns[t].resample("W").last().dropna()
            if series_weekly.empty:
                weekly_signals[t] = pd.Series(dtype=float)
            else:
                weekly_signals[t] = compute_stateful_signal_with_decay(
                    series_weekly, weekly_params
                )
        weekly_signals_df = pd.concat(weekly_signals, axis=1).fillna(0)
        print(
            f"DEBUG: Weekly signals (last rows) for cluster {cluster_label}:\n",
            weekly_signals_df.tail(),
        )
        print("Weekly signals summary:\n", weekly_signals_df.describe())

        # We no longer re-run weight optimization; assume weights are already in group_params.
        print(
            f"Using cached weights: weight_daily={group_params['weight_daily']}, weight_weekly={group_params['weight_weekly']} for cluster {cluster_label}"
        )

        # Update the cache for each ticker in this cluster.
        for ticker in tickers_in_returns:
            global_cache[ticker] = group_params

        # Build the group_signal dictionary.
        group_signal = {
            "tickers": tickers_in_returns,
            "daily": daily_signals,  # dict: ticker -> pd.Series
            "weekly": weekly_signals,  # dict: ticker -> pd.Series
        }
        print(f"DEBUG: group_signal for cluster {cluster_label}:", group_signal)

        # Flatten group_signal so that each ticker becomes its own key.
        if "tickers" in group_signal:
            for ticker in group_signal["tickers"]:
                overall_signals[ticker] = {
                    "daily": group_signal["daily"].get(ticker, {}),
                    "weekly": group_signal["weekly"].get(ticker, {}),
                }
        else:
            overall_signals.update(group_signal)
        print(f"Group {cluster_label}: {len(tickers_in_returns)} tickers optimized.")

    return overall_signals
