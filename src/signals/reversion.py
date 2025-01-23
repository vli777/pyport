import numpy as np

window_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def get_dynamic_thresholds(price_df, window=20, multiplier=1.0):
    """
    Calculate dynamic overbought and oversold thresholds based on historical volatility.

    Args:
        price_df (pd.DataFrame): DataFrame containing price data of tickers.
        window (int): Rolling window size.
        multiplier (float): Multiplier to adjust threshold based on volatility.

    Returns:
        float, float: Dynamic overbought and oversold thresholds.
    """
    z_scores = calculate_z_score(price_df, window=window)
    # Calculate standard deviation of Z-Scores across all tickers
    z_std = z_scores.std().mean()
    overbought_threshold = multiplier * z_std
    oversold_threshold = -multiplier * z_std
    return overbought_threshold, oversold_threshold


def find_optimal_window(
    price_series,
    test_windows=window_range,
    overbought_threshold=1.0,
    oversold_threshold=-1.0,
):
    """
    Find the optimal mean reversion window for a single asset based on historical data.

    Args:
        price_series (pd.Series): Price series of the asset.
        test_windows (list): List of rolling windows to test.
        overbought_threshold (float): Z-Score threshold for overbought condition.
        oversold_threshold (float): Z-Score threshold for oversold condition.

    Returns:
        int: Optimal rolling window size for the asset.
    """
    best_window = None
    best_reversion_score = -np.inf  # Initialize with a very low score

    for window in test_windows:
        # Calculate Z-scores for the current window
        z_scores = calculate_z_score(price_series.to_frame(), window=window)[
            price_series.name
        ]

        # Count reversion cases: Z-Score breaches and reversion within 1 std
        reversion_count = 0
        breach_count = 0
        for i in range(len(z_scores) - 1):
            if z_scores.iloc[i] > overbought_threshold:
                breach_count += 1
                if z_scores.iloc[i + 1] < overbought_threshold:
                    reversion_count += 1
            elif z_scores.iloc[i] < oversold_threshold:
                breach_count += 1
                if z_scores.iloc[i + 1] > oversold_threshold:
                    reversion_count += 1

        # Calculate the reversion score
        reversion_score = reversion_count / breach_count if breach_count > 0 else 0

        # Update the best window if the score improves
        if reversion_score > best_reversion_score:
            best_reversion_score = reversion_score
            best_window = window

    return best_window


def find_dynamic_windows(
    price_df,
    test_windows=window_range,
    overbought_threshold=1.0,
    oversold_threshold=-1.0,
):
    """
    Find optimal mean reversion windows for each asset in a DataFrame.

    Args:
        price_df (pd.DataFrame): DataFrame containing price data of multiple tickers.
        test_windows (list): List of rolling windows to test.
        overbought_threshold (float): Z-Score threshold for overbought condition.
        oversold_threshold (float): Z-Score threshold for oversold condition.

    Returns:
        dict: Optimal rolling window size for each ticker.
    """
    optimal_windows = {}

    for ticker in price_df.columns:
        optimal_window = find_optimal_window(
            price_series=price_df[ticker],
            test_windows=test_windows,
            overbought_threshold=overbought_threshold,
            oversold_threshold=oversold_threshold,
        )
        optimal_windows[ticker] = optimal_window

    return optimal_windows


def calculate_z_score(price_df, window=20):
    """
    Calculate the Z-Score of each ticker's returns over a rolling window.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns of tickers.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: Z-Score DataFrame.
    """
    rolling_mean = price_df.rolling(window=window).mean()
    rolling_std = price_df.rolling(window=window).std()
    z_score = (price_df - rolling_mean) / rolling_std
    return z_score


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
    dynamic_windows,
    dynamic_thresholds_fn=get_dynamic_thresholds,
    multiplier=1.0,
    plot=False,
):
    """
    Apply mean reversion strategy with dynamic windows and thresholds.

    Args:
        price_df (pd.DataFrame): DataFrame containing prices of tickers.
        dynamic_windows (dict): Optimal rolling window size for each ticker.
        dynamic_thresholds_fn (callable): Function to calculate dynamic thresholds.
        multiplier (float): Multiplier for threshold adjustment.
        plot (bool): Whether to plot Z-Scores for visualization.

    Returns:
        List[str], List[str]: Lists of ticker symbols to exclude and include respectively.
    """
    tickers_to_exclude = []
    tickers_to_include = []

    for ticker in price_df.columns:
        # Use the dynamic window for the current ticker
        window = dynamic_windows.get(ticker, 20)  # Default to 20 if not available

        # Calculate Z-Scores
        z_scores = calculate_z_score(price_df[[ticker]], window=window)

        # Calculate dynamic thresholds
        overbought_threshold, oversold_threshold = dynamic_thresholds_fn(
            price_df[[ticker]], window=window, multiplier=multiplier
        )

        # Generate mean reversion signals
        overextended, oversold = generate_mean_reversion_signals(
            z_scores,
            overbought_threshold=overbought_threshold,
            oversold_threshold=oversold_threshold,
        )

        # Append tickers to respective lists
        if overextended.iloc[-1]:
            tickers_to_exclude.append(ticker)
        if oversold.iloc[-1]:
            tickers_to_include.append(ticker)

        # Optional: Plot Z-Scores
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(14, 7))
            plt.plot(z_scores[ticker], label=f"{ticker} Z-Score")
            plt.axhline(
                y=overbought_threshold, color="r", linestyle="--", label="Overbought"
            )
            plt.axhline(
                y=oversold_threshold, color="g", linestyle="--", label="Oversold"
            )
            plt.title(f"Z-Score of {ticker} for Mean Reversion")
            plt.legend()
            plt.show()

    return tickers_to_exclude, tickers_to_include
