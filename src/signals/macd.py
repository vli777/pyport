def calculate_macd(price_df, fast=12, slow=26, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for each ticker.

    Args:
        price_df (pd.DataFrame): DataFrame containing price data of tickers.
        fast (int): Fast EMA window.
        slow (int): Slow EMA window.
        signal (int): Signal line window.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: MACD, MACD Signal, MACD Histogram DataFrames.
    """
    macd = price_df.ta.macd(fast=fast, slow=slow, signal=signal, append=False)
    macd_line = macd[f"MACD_{fast}_{slow}_{signal}"]
    macd_signal = macd[f"MACDs_{fast}_{slow}_{signal}"]
    macd_hist = macd[f"MACDh_{fast}_{slow}_{signal}"]
    return macd_line, macd_signal, macd_hist


def generate_macd_crossovers(macd_line, macd_signal):
    """
    Generate MACD crossover signals.

    Args:
        macd_line (pd.Series): MACD Line Series.
        macd_signal (pd.Series): MACD Signal Line Series.

    Returns:
        pd.Series, pd.Series: Boolean Series indicating bullish and bearish crossovers.
    """
    bullish = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    bearish = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
    bullish = bullish.fillna(False)
    bearish = bearish.fillna(False)
    return bullish, bearish


def generate_macd_preemptive_signals(
    macd_line, smoothing_window=5, roc_threshold=0.01, plateau_period=3
):
    """
    Generate preemptive signals based on the plateauing of the MACD.

    Args:
        macd_line (pd.Series): MACD Line Series.
        smoothing_window (int): Window size for smoothing the MACD line.
        roc_threshold (float): Threshold for the rate of change to consider as plateau.
        plateau_period (int): Number of consecutive periods the ROC must be below the threshold.

    Returns:
        pd.Series: Boolean Series indicating preemptive signals.
    """
    # Smooth the MACD line using EMA
    smoothed_macd = macd_line.ewm(span=smoothing_window, adjust=False).mean()

    # Calculate Rate of Change (ROC)
    roc = smoothed_macd.pct_change()

    # Identify periods where ROC is below the threshold (indicating plateau)
    plateau = roc.abs() < roc_threshold

    # Check if there has been a plateau for the specified number of consecutive periods
    plateau_signal = plateau.rolling(window=plateau_period).sum() == plateau_period

    # Forward fill to extend the plateau signal until a new significant change occurs
    plateau_signal = plateau_signal.fillna(False)

    return plateau_signal
