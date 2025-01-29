from typing import Dict, List, Tuple

import pandas as pd
from reversion.apply_mean_reversion import apply_mean_reversion
from utils.portfolio_utils import resample_returns


def apply_mean_reversion_multiscale(
    returns_df: pd.DataFrame,
    test_windows: range = range(10, 101, 10),
    plot: bool = False,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, int]]]:
    """
    Apply mean reversion strategy across multiple time scales with dynamic rolling windows.

    Args:
        returns_df (pd.DataFrame): Daily log returns DataFrame.
        test_windows (range, optional): Rolling windows to test.
        plot (bool): Whether to plot Z-Scores.
        n_jobs (int): Number of parallel jobs.

    Returns:
        Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, int]]]:
            - Signals structured by time scale.
            - Dynamic rolling windows per time scale and ticker.
    """
    resampled = resample_returns(returns_df)
    signals = {}
    windows = {}

    # Apply on daily data
    recommendations_daily, optimal_windows_daily = apply_mean_reversion(
        returns_df=returns_df,
        test_windows=test_windows,
        plot=plot,
        n_jobs=n_jobs,
    )

    # Apply on weekly data
    recommendations_weekly, optimal_windows_weekly = apply_mean_reversion(
        returns_df=resampled["weekly"],
        test_windows=range(1, 52),
        plot=plot,
        n_jobs=n_jobs,
    )

    signals = {
        "daily": {
            "exclude": recommendations_daily["exclude"],
            "include": recommendations_daily["include"],
        },
        "weekly": {
            "exclude": recommendations_weekly["exclude"],
            "include": recommendations_weekly["include"],
        },
    }

    windows = {"daily": optimal_windows_daily, "weekly": optimal_windows_weekly}

    return signals, windows
