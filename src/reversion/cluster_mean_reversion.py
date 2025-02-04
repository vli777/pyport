import numpy as np
import pandas as pd

from reversion.optimize_window_threshold import optimize_robust_mean_reversion
from reversion.reversion_utils import cluster_stocks, group_ticker_params_by_cluster
from reversion.optimize_period_weights import find_optimal_weights


def cluster_mean_reversion(
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    global_cache: dict = None,  # Global cache passed in.
):
    if global_cache is None:
        global_cache = {}

    clusters = cluster_stocks(returns_df)
    print(f"{len(clusters)} clusters found")

    # Create a mapping from cluster -> known parameters (from previous runs)
    known_group_params = group_ticker_params_by_cluster(global_cache)

    # This will hold the group signals in the desired format.
    group_signals = {}

    for label, tickers in clusters.items():
        group_returns: pd.DataFrame = returns_df[tickers].dropna(how="all", axis=1)
        if group_returns.empty:
            continue

        # Try to find an existing group from the cache.
        existing_group_id = None
        existing_params = None
        for ticker in tickers:
            for cluster_id, group_data in known_group_params.items():
                if ticker in group_data["tickers"]:
                    existing_group_id = cluster_id
                    existing_params = group_data["params"]
                    break
            if existing_group_id:
                break  # Found at least one match

        if existing_params:
            # Apply existing parameters to all tickers in this group.
            for ticker in tickers:
                global_cache[ticker] = existing_params

            # We must also compute daily and weekly signals using these parameters.
            group_signal = compute_group_signals(
                group_returns=group_returns, tickers=tickers, params=existing_params
            )
            group_signals[label] = group_signal
            continue  # Skip re-optimization

        # If no existing parameters found, optimize the group.
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

        # Optimize the combination weights for daily and weekly signals.
        daily_signals_df = pd.DataFrame.from_dict(
            {
                t: group_returns[t].to_dict()
                for t in tickers
                if t in group_returns.columns
            },
            orient="index",
        ).T.fillna(0)
        # Here you might want to compute true weekly signals; for now we copy daily.
        weekly_signals_df = daily_signals_df.copy()

        group_weights = find_optimal_weights(
            daily_signals_df=daily_signals_df,
            weekly_signals_df=weekly_signals_df,
            returns_df=group_returns,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        # Combine parameters into one dict for this cluster.
        group_params = {
            "window_daily": round(best_params_daily.get("window", 20), 1),
            "z_threshold_daily": round(best_params_daily.get("z_threshold", 1.5), 1),
            "window_weekly": round(best_params_weekly.get("window", 5), 1),
            "z_threshold_weekly": round(best_params_weekly.get("z_threshold", 1.5), 1),
            "weight_daily": round(group_weights.get("weight_daily", 0.7), 1),
            "weight_weekly": round(group_weights.get("weight_weekly", 0.3), 1),
            "cluster": label,  # Store the current label
        }

        # Update the global cache for all tickers in this group.
        for ticker in tickers:
            global_cache[ticker] = group_params

        # Compute group signals (daily and weekly) using the optimized parameters.
        group_signal = compute_group_signals(
            group_returns=group_returns, tickers=tickers, params=group_params
        )
        group_signals[label] = group_signal

        print(f"Group {label}: {len(tickers)} tickers optimized.")

    return group_signals


def compute_group_signals(
    group_returns: pd.DataFrame, tickers: list, params: dict
) -> dict:
    """
    Given a group of tickers and their returns along with optimized parameters,
    compute the continuous daily and weekly signals for each ticker.

    The daily signal for a ticker is computed as the rolling z-score (using the
    specified window and threshold) for its returns. Similarly for the weekly signal.

    Returns a dictionary with keys:
       - "tickers": the list of tickers
       - "daily": dict mapping ticker -> {date: signal}
       - "weekly": dict mapping ticker -> {date: signal}
    """
    window_daily = int(params.get("window_daily", 20))
    z_threshold_daily = params.get("z_threshold_daily", 1.5)
    window_weekly = int(params.get("window_weekly", 5))
    z_threshold_weekly = params.get("z_threshold_weekly", 1.5)

    daily_signals = {}
    weekly_signals = {}

    # Compute daily signals.
    for ticker in tickers:
        series = group_returns[ticker].dropna()
        if series.empty:
            daily_signals[ticker] = {}
        else:
            # Daily signals
            rolling_mean = series.rolling(
                window=window_daily, min_periods=window_daily
            ).mean()
            rolling_std = series.rolling(
                window=window_daily, min_periods=window_daily
            ).std()
            z_scores = (series - rolling_mean) / (rolling_std.replace(0, np.nan))
            # Only keep signals that exceed the threshold; otherwise, set to 0.
            z_scores = z_scores.where(z_scores.abs() >= z_threshold_daily, 0)
            daily_signals[ticker] = z_scores.dropna().to_dict()

    # Compute weekly signals.
    # First, resample group returns to weekly frequency.
    weekly_returns = group_returns.resample("W").last()
    for ticker in tickers:
        series = weekly_returns[ticker].dropna()
        if series.empty:
            weekly_signals[ticker] = {}
        else:
            # Weekly signals
            rolling_mean = series.rolling(
                window=window_weekly, min_periods=window_weekly
            ).mean()
            rolling_std = series.rolling(
                window=window_weekly, min_periods=window_weekly
            ).std()
            z_scores = (series - rolling_mean) / (rolling_std.replace(0, np.nan))
            z_scores = z_scores.where(z_scores.abs() >= z_threshold_weekly, 0)
            weekly_signals[ticker] = z_scores.dropna().to_dict()

    return {
        "tickers": tickers,
        "daily": daily_signals,
        "weekly": weekly_signals,
    }
