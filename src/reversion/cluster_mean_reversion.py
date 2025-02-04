import pandas as pd

from reversion.optimize_window_threshold import optimize_robust_mean_reversion
from reversion.reversion_utils import cluster_stocks
from reversion.optimize_period_weights import find_optimal_weights


def cluster_mean_reversion(
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    reoptimize: bool = False,
    global_cache: dict = None,  # Global cache passed in.
):
    # global_cache should be a dict that persists across clusters.
    if global_cache is None:
        global_cache = {}

    clusters = cluster_stocks(returns_df)
    group_parameters = {}

    for label, tickers in clusters.items():
        group_returns = returns_df[tickers].dropna(how="all", axis=1)
        if group_returns.empty:
            continue

        # Check for missing tickers in the global cache.
        tickers_missing = [t for t in tickers if t not in global_cache]
        if tickers_missing:
            # Optimize robust mean reversion parameters for daily returns.
            best_params_daily, _ = optimize_robust_mean_reversion(
                group_returns,
                n_trials=n_trials,
                n_jobs=n_jobs,
                reoptimize=reoptimize,
                cache=global_cache,  # Using the same cache dict.
            )

            # Optimize for weekly returns.
            weekly_returns = group_returns.resample("W").last()
            best_params_weekly, _ = optimize_robust_mean_reversion(
                weekly_returns,
                n_trials=n_trials,
                n_jobs=n_jobs,
                reoptimize=reoptimize,
                cache=global_cache,
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
            # (For weekly signals, you would similarly build a dataframe from your weekly signal data.)
            # Here, for simplicity we assume the weekly signals are available in the same way.
            weekly_signals_df = (
                daily_signals_df.copy()
            )  # replace with actual weekly signals

            # Create a combined index for safety.
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
                reoptimize=reoptimize,
                cache=global_cache,
            )

            # Combine parameters into one dict for this cluster.
            group_params = {
                "window_daily": best_params_daily.get("window", 20),
                "z_threshold_daily": best_params_daily.get("z_threshold", 1.5),
                "window_weekly": best_params_weekly.get("window", 20),
                "z_threshold_weekly": best_params_weekly.get("z_threshold", 1.5),
                "weight_daily": group_weights.get("weight_daily", 0.5),
                "weight_weekly": group_weights.get("weight_weekly", 0.5),
            }

            # Update the global cache for all tickers in this group.
            for ticker in tickers:
                global_cache[ticker] = group_params

        # Save group-level parameters for potential downstream use.
        group_parameters[label] = {
            "tickers": tickers,
            "params": global_cache[tickers[0]],
        }
        print(f"Group {label}: {len(tickers)} tickers optimized.")

    return group_parameters
