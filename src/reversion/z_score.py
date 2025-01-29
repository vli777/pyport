import pandas as pd


def calculate_z_score(returns_df: pd.DataFrame, window: int = 20):
    """
    Calculate the Z-Score for each ticker in a returns DataFrame based on a rolling window.

    Args:
        returns_df (pd.DataFrame): DataFrame of log returns with tickers as columns and dates as index.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: Z-Score DataFrame for all tickers.
    """
    z_scores = returns_df.apply(
        lambda x: (x - x.rolling(window=window, min_periods=1).mean())
        / x.rolling(window=window, min_periods=1).std(),
        axis=0,
    )
    return z_scores
