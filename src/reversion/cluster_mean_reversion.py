import pandas as pd

from reversion.optimize_window_threshold import optimize_robust_mean_reversion
from reversion.reversion_utils import cluster_stocks
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

    group_parameters = {}

    for label, tickers in clusters.items():
        group_returns: pd.DataFrame = returns_df[tickers].dropna(how="all", axis=1)
        if group_returns.empty:
            continue

        # Use label as a string for the group id.
        group_id = str(label)

        # Check for missing tickers in the global cache.
        tickers_missing = [t for t in tickers if t not in global_cache]
        if tickers_missing:
            # Optimize robust mean reversion parameters for daily returns.
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
            combined_dates = daily_signals_df.index.union(
                weekly_signals_df.index
            ).union(group_returns.index)
            daily_signals_df = daily_signals_df.reindex(combined_dates).fillna(0)
            weekly_signals_df = weekly_signals_df.reindex(combined_dates).fillna(0)

            group_weights = find_optimal_weights(
                daily_signals_df=daily_signals_df,
                weekly_signals_df=weekly_signals_df,
                returns_df=group_returns,
                n_trials=n_trials,
                n_jobs=n_jobs,
                group_id=group_id,
            )

            # Combine parameters into one dict for this cluster.
            group_params = {
                "window_daily": round(best_params_daily.get("window", 20), 1),
                "z_threshold_daily": round(best_params_daily.get("z_threshold", 1.5), 1),
                "window_weekly": round(best_params_weekly.get("window", 5), 1),
                "z_threshold_weekly": round(best_params_weekly.get("z_threshold", 1.5), 1),
                "weight_daily": round(group_weights.get("weight_daily", 0.7), 1),
                "weight_weekly": round(group_weights.get("weight_weekly", 0.3), 1),
                "cluster": group_id,
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
