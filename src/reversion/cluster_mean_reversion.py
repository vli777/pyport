import pandas as pd

from reversion.optimize_window_threshold import optimize_robust_mean_reversion
from reversion.reversion_utils import cluster_stocks, compute_ticker_hash


def cluster_mean_reversion(
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
):
    clusters = cluster_stocks(returns_df)
    group_parameters = {}

    for label, tickers in clusters.items():
        group_returns = returns_df[tickers].dropna(how="all", axis=1)
        if group_returns.empty:
            continue

        # Construct unique cache filenames for daily and weekly parameters for this group
        ticker_hash = compute_ticker_hash(tickers)
        cache_filename_daily = f"{cache_dir}/reversion_{ticker_hash}_daily.pkl"
        cache_filename_weekly = f"{cache_dir}/reversion_{ticker_hash}_weekly.pkl"

        # Optimize daily signals for the group
        best_params_daily, _ = optimize_robust_mean_reversion(
            group_returns,
            n_trials=n_trials,
            n_jobs=n_jobs,
            cache_filename=cache_filename_daily,
            reoptimize=reoptimize,
        )

        # For weekly signals, resample returns (e.g., using last observation of the week)
        weekly_returns = group_returns.resample("W").last()
        best_params_weekly, _ = optimize_robust_mean_reversion(
            weekly_returns,
            n_trials=n_trials,
            n_jobs=n_jobs,
            cache_filename=cache_filename_weekly,
            reoptimize=reoptimize,
        )

        group_parameters[label] = {
            "tickers": tickers,
            "daily": best_params_daily,
            "weekly": best_params_weekly,
        }

        print(f"Group {label}: {len(tickers)} tickers optimized.")

    return group_parameters
