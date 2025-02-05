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

    group_parameters = {}

    for label, tickers in clusters.items():
        group_returns: pd.DataFrame = returns_df[tickers].dropna(how="all", axis=1)
        if group_returns.empty:
            continue

        # **New Method: Find an existing group that contains at least one of these tickers**
        existing_group_id = None
        existing_params = None
        for ticker in tickers:
            for cluster_id, group_data in known_group_params.items():
                if ticker in group_data["tickers"]:
                    existing_group_id = cluster_id
                    existing_params = group_data["params"]
                    break
            if existing_group_id:
                break  # Stop checking once we find a match

        if existing_params:
            # Apply existing parameters to all tickers in this group
            for ticker in tickers:
                global_cache[ticker] = existing_params

            group_parameters[label] = {
                "tickers": tickers,
                "params": existing_params,
            }
            continue  # Skip re-optimization

        # **If no existing parameters found, optimize the group.**
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
        weekly_signals_df = (
            daily_signals_df.copy()
        )  # Replace with actual weekly signals if available.

        combined_dates = daily_signals_df.index.union(weekly_signals_df.index).union(
            group_returns.index
        )
        daily_signals_df = daily_signals_df.reindex(combined_dates).fillna(0)
        weekly_signals_df = weekly_signals_df.reindex(combined_dates).fillna(0)

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
            "cluster": label,  # Store as the current label
        }

        # Update the global cache for all tickers in this group.
        for ticker in tickers:
            global_cache[ticker] = group_params

        group_parameters[label] = {
            "tickers": tickers,
            "params": global_cache[tickers[0]],
        }
        print(f"Group {label}: {len(tickers)} tickers optimized.")

    return group_parameters
