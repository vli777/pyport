from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import Float


from reversion.reversion_window import manage_dynamic_windows
from reversion.z_score import get_zscore_thresholds_ticker, plot_z_scores_grid
from reversion.zscore_multiplier import optimize_multiplier
from utils.portfolio_utils import resample_returns
from utils import logger


def generate_mean_reversion_signals(
    z_score_df: pd.DataFrame, dynamic_thresholds: Dict[str, Tuple[Float, Float]]
) -> Dict[str, Dict[str, bool]]:
    """
    Generate mean reversion signals based on Z-Score and dynamic thresholds.

    Args:
        z_score_df (pd.DataFrame): Z-Score DataFrame with tickers as columns and dates as index.
        dynamic_thresholds (dict): {ticker: (overbought_threshold, oversold_threshold)}

    Returns:
        dict: {ticker: {'overextended': bool, 'oversold': bool}}
    """
    signals = {}

    # Iterate over each ticker to generate signals
    for ticker in z_score_df.columns:
        if ticker not in dynamic_thresholds:
            continue

        # Get the latest Z-score and thresholds
        latest_z = z_score_df[ticker].iloc[-1]
        overbought_threshold, oversold_threshold = dynamic_thresholds[ticker]

        # Determine signals
        overextended = latest_z > overbought_threshold
        oversold = latest_z < oversold_threshold

        # Store results in dictionary
        signals[ticker] = {"overextended": overextended, "oversold": oversold}

    return signals


def apply_mean_reversion(
    returns_df: pd.DataFrame,
    test_windows: range = range(10, 101, 10),
    plot: bool = False,
    n_jobs: int = -1,
) -> Tuple[List[str], List[str]]:
    """
    Apply mean reversion strategy with dynamic rolling windows based on log returns.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame with tickers as columns and dates as index.
        test_windows (range, optional): Rolling windows to test during discovery.
        plot (bool): Whether to plot Z-Scores for visualization.
        n_jobs (int): Number of parallel jobs to run. -1 utilizes all available CPU cores.

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, int]]:
            - Signals structured by time scale.
            - Dynamic rolling windows per ticker.

    """
    logger.info("Discovering optimal rolling windows for each ticker...")
    dynamic_windows = manage_dynamic_windows(
        returns_df=returns_df,
        test_windows=test_windows,
        overbought_threshold=1.0,
        oversold_threshold=-1.0,
        n_jobs=n_jobs,
    )
    logger.info("Optimal rolling windows discovered.")

    # Optimize multipliers using the initial window (could iterate per ticker if needed)
    optimal_multipliers = optimize_multiplier(
        returns_df=returns_df, window=test_windows.start, n_trials=50
    )
    overbought_multiplier = optimal_multipliers["overbought_multiplier"]
    oversold_multiplier = optimal_multipliers["oversold_multiplier"]
    logger.info(
        f"Optimal multipliers determined: {overbought_multiplier}, {oversold_multiplier}"
    )

    # Calculate Z-score thresholds using optimized multipliers and window
    # For multiplier optimization, use a single window per ticker if dynamic_windows allows
    dynamic_thresholds = get_zscore_thresholds_ticker(
        returns_df,
        window=test_windows.start,  # Assuming window is a single int here
        overbought_multiplier=overbought_multiplier,
        oversold_multiplier=oversold_multiplier,
    )

    # Calculate Z-scores
    z_scores_dict = {}
    for ticker in returns_df.columns:
        window = test_windows.start  # Using the optimized window
        rolling_mean = returns_df[ticker].rolling(window=window, min_periods=1).mean()
        rolling_std = returns_df[ticker].rolling(window=window, min_periods=1).std()
        z_scores_dict[ticker] = (
            returns_df[ticker] - rolling_mean
        ) / rolling_std.replace(0, np.nan)

    z_score_df = pd.DataFrame(z_scores_dict)

    # Generate signals
    signals = generate_mean_reversion_signals(z_score_df, dynamic_thresholds)

    # Separate tickers based on signals
    tickers_to_exclude = [
        ticker for ticker, signal in signals.items() if signal["overextended"]
    ]
    tickers_to_include = [
        ticker for ticker, signal in signals.items() if signal["oversold"]
    ]

    logger.info(f"Mean Reversion Overbought: {tickers_to_exclude}")
    logger.info(f"Mean Reversion Oversold: {tickers_to_include}")

    # Optional: Plot Z-Scores
    if plot:
        plot_z_scores_grid(
            z_scores_df=z_score_df,
            overbought_thresholds=pd.Series(
                {k: v[0] for k, v in dynamic_thresholds.items()}
            ),
            oversold_thresholds=pd.Series(
                {k: v[1] for k, v in dynamic_thresholds.items()}
            ),
            grid_shape=(6, 6),
            figsize=(24, 24),
        )

    return (
        {
            "daily": {
                "exclude": tickers_to_exclude,  # Adjust if separate for daily
                "include": tickers_to_include,  # Adjust if separate for daily
            },
            "weekly": {
                "exclude": tickers_to_exclude,  # Adjust if separate for weekly
                "include": tickers_to_include,  # Adjust if separate for weekly
            },
        },
        dynamic_windows,  # Return dynamic_windows
    )


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
        test_windows=test_windows,
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
