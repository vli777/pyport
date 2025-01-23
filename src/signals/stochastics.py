def calculate_stochastic_full(
    price_df, windows=[5, 10, 15, 20, 25, 30], smooth_k=3, smooth_d=3
):
    """
    Calculate the weighted average of StochasticFull indicators across multiple windows.

    Args:
        price_df (pd.DataFrame): DataFrame containing price data of tickers.
        windows (list): List of window intervals for %K and %D calculation.
        smooth_k (int): Smoothing period for %K.
        smooth_d (int): Smoothing period for %D.

    Returns:
        pd.DataFrame: DataFrame with weighted average of %K and %D for each ticker.
    """
    weight_sum = sum(range(1, len(windows) + 1))  # Calculate total weight
    weighted_k = 0
    weighted_d = 0

    for i, window in enumerate(windows):
        # Calculate StochasticFull for each window
        stoch = price_df.ta.stoch(
            length=window, smooth_k=smooth_k, smooth_d=smooth_d, append=False
        )
        stoch_k = stoch[f"STOCHK_{window}_{smooth_k}_{smooth_d}"]
        stoch_d = stoch[f"STOCHD_{window}_{smooth_k}_{smooth_d}"]

        # Apply weights (e.g., 1 for smallest window, 2 for next, etc.)
        weight = i + 1
        weighted_k += stoch_k * weight
        weighted_d += stoch_d * weight

    # Compute weighted average
    avg_k = weighted_k / weight_sum
    avg_d = weighted_d / weight_sum

    return avg_k, avg_d


def generate_convergence_signal(
    stoch_k, stoch_d, overbought=80, oversold=20, tolerance=5
):
    """
    Generate buy/sell signals when all window intervals converge at overbought or oversold levels.

    Args:
        stoch_k (pd.Series): Weighted average %K.
        stoch_d (pd.Series): Weighted average %D.
        overbought (float): Overbought threshold.
        oversold (float): Oversold threshold.
        tolerance (float): Tolerance level for convergence.

    Returns:
        bool, bool: Buy and sell signals.
    """
    # Calculate the difference between %K and %D
    diff = (stoch_k - stoch_d).abs()

    # Check for convergence
    convergence = diff <= tolerance

    # Generate buy/sell signals based on overbought/oversold levels
    buy_signal = convergence & (stoch_k < oversold) & (stoch_d < oversold)
    sell_signal = convergence & (stoch_k > overbought) & (stoch_d > overbought)

    return buy_signal.iloc[-1], sell_signal.iloc[-1]
