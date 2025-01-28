from joblib import Parallel, delayed
import numpy as np

from utils import logger


def manage_dynamic_windows(
    returns_df, test_windows, overbought_threshold, oversold_threshold, n_jobs=-1
):
    """
    Load or calculate dynamic windows for mean reversion using returns data.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame with tickers as columns and dates as index.
        test_windows (iterable): Range of rolling windows to test.
        overbought_threshold (float): Z-Score threshold for overbought condition.
        oversold_threshold (float): Z-Score threshold for oversold condition.
        n_jobs (int): Number of jobs to run in parallel. -1 uses all available processors.

    Returns:
        dict: {ticker: optimal_window}
    """
    dynamic_windows = find_dynamic_windows(
        returns_df=returns_df,
        test_windows=test_windows,
        overbought_threshold=overbought_threshold,
        oversold_threshold=oversold_threshold,
        n_jobs=n_jobs,
    )
    return dynamic_windows


def find_optimal_window(
    returns_series,
    test_windows=range(10, 101, 10),
    overbought_threshold=1.0,
    oversold_threshold=-1.0,
):
    best_window = None
    best_reversion_score = -np.inf  # Initialize with a very low score

    for window in test_windows:
        if len(returns_series) < window:
            # Not enough data for this window
            logger.warning(f"Insufficient data for window {window} in series.")
            continue  # Skip to the next window

        # Calculate Z-scores for the current window
        rolling_mean = returns_series.rolling(window).mean()
        rolling_std = returns_series.rolling(window).std()

        # Drop NaN values resulting from insufficient data
        valid = rolling_std.notna()
        if not valid.any():
            logger.warning(f"All Z-scores are NaN for window {window}. Skipping.")
            continue

        z_scores = (returns_series - rolling_mean) / rolling_std

        # Detect breaches
        overbought = z_scores > overbought_threshold
        oversold = z_scores < oversold_threshold
        breach = overbought | oversold

        # Detect reversion by shifting the signals by one period
        reversion = (overbought & (z_scores.shift(-1) < overbought_threshold)) | (
            oversold & (z_scores.shift(-1) > oversold_threshold)
        )

        # Calculate counts
        breach_count = breach.sum()
        reversion_count = reversion.sum()

        # Calculate the reversion score
        reversion_score = reversion_count / breach_count if breach_count > 0 else 0

        # Update the best window if the score improves
        if reversion_score > best_reversion_score:
            best_reversion_score = reversion_score
            best_window = window

    return best_window


def find_dynamic_windows(
    returns_df,
    test_windows=range(10, 101, 10),
    overbought_threshold=1.0,
    oversold_threshold=-1.0,
    n_jobs=-1,  # Use all available CPU cores
):
    """
    Find optimal mean reversion windows for each ticker using returns data and parallel processing.
    """
    def process_ticker(ticker):
        optimal_window = find_optimal_window(
            returns_series=returns_df[ticker],
            test_windows=test_windows,
            overbought_threshold=overbought_threshold,
            oversold_threshold=oversold_threshold,
        )
        return (ticker, optimal_window)

    # Parallel processing of tickers to find optimal windows
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_ticker)(ticker) for ticker in returns_df.columns
    )

    # Convert list of tuples to dictionary
    optimal_windows = dict(results)

    return optimal_windows
