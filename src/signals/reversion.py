import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd

from utils import logger
from signals.z_score_plot import plot_z_scores_grid


def calculate_z_score(price_series, window):
    """
    Calculate the Z-Score for a given price series based on a rolling window.

    Args:
        price_series (pd.Series): Series of 'Adj Close' prices for a ticker.
        window (int): Rolling window size.

    Returns:
        pd.Series: Z-Score series.
    """
    rolling_mean = price_series.rolling(window=window, min_periods=1).mean()
    rolling_std = price_series.rolling(window=window, min_periods=1).std()
    z_scores = (price_series - rolling_mean) / rolling_std
    return z_scores


def find_optimal_window(
    price_series,
    test_windows=range(10, 101, 10),
    overbought_threshold=1.0,
    oversold_threshold=-1.0,
):
    """
    Find the optimal mean reversion window for a single asset based on historical data using vectorized operations.

    Args:
        price_series (pd.Series): Price series of the asset.
        test_windows (iterable): List of rolling windows to test.
        overbought_threshold (float): Z-Score threshold for overbought condition.
        oversold_threshold (float): Z-Score threshold for oversold condition.

    Returns:
        int: Optimal rolling window size for the asset.
    """
    best_window = None
    best_reversion_score = -np.inf  # Initialize with a very low score

    for window in test_windows:
        # Calculate Z-scores for the current window
        # Assuming calculate_z_score can accept a Series directly
        z_scores = calculate_z_score(price_series, window=window)

        # Detect breaches
        overbought = z_scores > overbought_threshold
        oversold = z_scores < oversold_threshold
        breach = overbought | oversold

        # Detect reversion by shifting the signals by one day
        # Added parentheses to ensure correct operator precedence
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
    price_df,
    test_windows=range(10, 101, 10),
    overbought_threshold=1.0,
    oversold_threshold=-1.0,
    n_jobs=-1,  # Use all available CPU cores
):
    """
    Find optimal mean reversion windows for each ticker using parallel processing.

    Args:
        price_df (pd.DataFrame): Multi-level Price data of multiple tickers (columns=(ticker, field), index=dates).
        test_windows (iterable): Possible rolling windows to test.
        overbought_threshold (float): Fixed threshold for overbought detection during testing.
        oversold_threshold (float): Fixed threshold for oversold detection during testing.
        n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.

    Returns:
        dict: {ticker: optimal_window}
    """
    # Extract only 'Adj Close' columns
    try:
        adj_close_df = price_df.xs("Adj Close", level=1, axis=1)
    except KeyError:
        logger.error("The 'Adj Close' column is missing from the price DataFrame.")
        raise

    def process_ticker(ticker):
        optimal_window = find_optimal_window(
            price_series=adj_close_df[ticker],
            test_windows=test_windows,
            overbought_threshold=overbought_threshold,
            oversold_threshold=oversold_threshold,
        )
        return (ticker, optimal_window)

    # Parallel processing of tickers to find optimal windows
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_ticker)(ticker) for ticker in adj_close_df.columns
    )

    # Convert list of tuples to dictionary
    optimal_windows = dict(results)

    return optimal_windows


def get_dynamic_thresholds(price_df, window=20, multiplier=1.0):
    """
    Calculate ticker-specific dynamic overbought and oversold thresholds based on volatility of Z-scores.

    Args:
        price_df (pd.DataFrame): Single ticker DataFrame or slice (columns=[ticker]).
        window (int): Rolling window size for Z-score calculation.
        multiplier (float): Scales the standard deviation.

    Returns:
        (float, float): Overbought threshold, Oversold threshold
    """
    z_scores = calculate_z_score(price_df, window=window)
    z_std = z_scores.std().iloc[0]  # single ticker => single column => .iloc[0]

    overbought_threshold = multiplier * z_std
    oversold_threshold = -multiplier * z_std

    return overbought_threshold, oversold_threshold


def generate_mean_reversion_signals(
    z_score_df, overbought_threshold=1.0, oversold_threshold=-1.0
):
    """
    Generate mean reversion signals based on Z-Score.

    Args:
        z_score_df (pd.DataFrame): Z-Score DataFrame.
        overbought_threshold (float): Z-Score threshold to identify overextended stocks.
        oversold_threshold (float): Z-Score threshold to identify undervalued stocks.

    Returns:
        pd.Series, pd.Series: Boolean Series indicating overextended and oversold tickers respectively.
    """
    latest_z = z_score_df.iloc[-1]
    # Identify overbought tickers
    overextended = latest_z > overbought_threshold
    # Identify oversold tickers
    oversold = latest_z < oversold_threshold
    # Handle NaNs by treating them as False
    overextended = overextended.fillna(False)
    oversold = oversold.fillna(False)
    return overextended, oversold


def apply_mean_reversion(
    price_df,
    test_windows=range(10, 101, 10),  # Range of rolling windows to test
    multiplier=1.0,
    plot=False,
    n_jobs=-1,  # Number of parallel jobs; -1 uses all available cores
):
    """
    Apply mean reversion strategy with dynamic windows and fixed thresholds using only 'Adj Close' prices.

    Args:
        price_df (pd.DataFrame): Multi-level DataFrame containing 'Adj Close' prices of tickers.
                                 Columns are tuples like (TICKER, 'Adj Close').
                                 Index = dates.
        test_windows (iterable, optional): Range of rolling windows to test during discovery.
        multiplier (float): Multiplier for threshold adjustment after discovery.
        plot (bool): Whether to plot Z-Scores for visualization.
        n_jobs (int): Number of parallel jobs to run. -1 utilizes all available CPU cores.

    Returns:
        Tuple[List[str], List[str]]:
            - List of ticker symbols to exclude (overbought).
            - List of ticker symbols to include (oversold).
    """
    # Extract only 'Adj Close' columns
    try:
        adj_close_df = price_df.xs("Adj Close", level=1, axis=1)
    except KeyError:
        logger.error("The 'Adj Close' column is missing from the price DataFrame.")
        raise

    # Discover optimal windows using fixed thresholds
    logger.info("Discovering optimal rolling windows for each ticker...")
    dynamic_windows = find_dynamic_windows(
        price_df=price_df,  # Passing multi-level DataFrame
        test_windows=test_windows,
        overbought_threshold=1.0,  # Fixed overbought threshold
        oversold_threshold=-1.0,  # Fixed oversold threshold
    )
    logger.info("Optimal rolling windows discovered.")

    # Initialize dictionaries to store Z-Scores and thresholds
    z_scores_dict = {}
    overbought_thresholds = {}
    oversold_thresholds = {}

    def process_ticker(ticker):
        window = dynamic_windows.get(ticker, 20)  # Default window size 20 if not found
        price_series = adj_close_df[ticker]

        # Calculate Z-Scores
        z_scores = calculate_z_score(price_series, window=window)

        # Determine dynamic thresholds based on multiplier
        # Assuming dynamic_thresholds_fn is a fixed function that applies the multiplier
        overbought_threshold = 1.0 * multiplier
        oversold_threshold = -1.0 * multiplier

        # Store Z-Scores and thresholds
        z_scores_dict[ticker] = z_scores
        overbought_thresholds[ticker] = overbought_threshold
        oversold_thresholds[ticker] = oversold_threshold

        # Determine if latest Z-Score breaches thresholds
        latest_z = z_scores.iloc[-1]
        overextended = latest_z > overbought_threshold
        is_oversold = latest_z < oversold_threshold

        return (ticker, overextended, is_oversold)

    logger.info("Starting parallel processing of tickers for mean reversion signals...")
    # Parallel processing of tickers
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_ticker)(ticker) for ticker in adj_close_df.columns
    )
    logger.info("Parallel processing completed.")

    # Separate tickers to exclude and include based on signals
    tickers_to_exclude = [ticker for ticker, overext, _ in results if overext]
    tickers_to_include = [ticker for ticker, _, oversold in results if oversold]

    logger.info(f"Tickers to Exclude (Overbought): {tickers_to_exclude}")
    logger.info(f"Tickers to Include (Oversold): {tickers_to_include}")

    # Optional: Plot Z-Scores
    if plot:
        # Create DataFrames from the dictionaries
        z_scores_df = pd.DataFrame(z_scores_dict)
        overbought_series = pd.Series(overbought_thresholds)
        oversold_series = pd.Series(oversold_thresholds)

        # Call the plotting function
        plot_z_scores_grid(
            z_scores_df=z_scores_df,
            overbought_thresholds=overbought_series,
            oversold_thresholds=oversold_series,
            grid_shape=(6, 6),
            figsize=(24, 24),  # Adjusted for better readability
        )

    return tickers_to_exclude, tickers_to_include
