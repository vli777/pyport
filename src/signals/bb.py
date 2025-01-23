def calculate_bollinger_bands(price_df, window=20, window_dev=2):
    """
    Calculate Bollinger Bands for each ticker.

    Args:
        price_df (pd.DataFrame): DataFrame containing price data of tickers.
        window (int): Rolling window size.
        window_dev (int): Number of standard deviations.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Upper Band, Middle Band, Lower Band DataFrames.
    """
    bollinger = price_df.ta.bbands(length=window, std=window_dev, append=False)
    upper_band = bollinger[f"BBU_{window}_{window_dev}"]
    middle_band = bollinger[f"BBM_{window}_{window_dev}"]
    lower_band = bollinger[f"BBL_{window}_{window_dev}"]
    return upper_band, middle_band, lower_band


def generate_bollinger_signals(price_df, upper_band, lower_band):
    """
    Generate Bollinger Bands signals for overbought and oversold conditions.

    Args:
        price_df (pd.DataFrame): Price DataFrame.
        upper_band (pd.DataFrame): Upper Bollinger Band DataFrame.
        lower_band (pd.DataFrame): Lower Bollinger Band DataFrame.

    Returns:
        pd.Series, pd.Series: Boolean Series indicating overbought and oversold tickers.
    """
    latest_price = price_df.iloc[-1]
    latest_upper = upper_band.iloc[-1]
    latest_lower = lower_band.iloc[-1]

    bollinger_overbought = latest_price > latest_upper
    bollinger_oversold = latest_price < latest_lower

    bollinger_overbought = bollinger_overbought.fillna(False)
    bollinger_oversold = bollinger_oversold.fillna(False)

    return bollinger_overbought, bollinger_oversold
