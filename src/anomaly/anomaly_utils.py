import pandas as pd


def detect_meme_stocks(
    df: pd.DataFrame, lookback_days: int = 90, crash_days: int = 30
) -> set:
    """
    Identify meme/manipulated stocks based on extreme price increases and sudden crashes.

    Args:
        df (pd.DataFrame): DataFrame with stock prices (columns = stock tickers, rows = dates).
        lookback_days (int): Number of days to check for extreme price jumps.
        crash_days (int): Number of days to check for post-spike crashes.

    Returns:
        set: A set of stock tickers classified as meme/manipulated stocks.
    """
    # Calculate price multiples over the lookback period (e.g., 3 months)
    price_multiple = df.iloc[-1] / df.shift(lookback_days).iloc[-1]

    # Threshold: Any stock in the **top 1% percentile** for price increase
    threshold = price_multiple.quantile(0.99)
    high_flyers = set(price_multiple[price_multiple > threshold].index)

    # Detect stocks that crashed **50%+ from their recent peak in the last crash_days**
    rolling_max = df.rolling(crash_days).max()
    drawdown = df.iloc[-1] / rolling_max.iloc[-1]
    crashing_stocks = set(drawdown[drawdown < 0.5].index)  # Stocks that lost 50%+

    # Combine both anomaly sets
    meme_candidates = high_flyers | crashing_stocks  # Union of sets
    return meme_candidates


def get_cache_filename(method: str) -> str:
    """Return the correct cache filename based on the anomaly detection method."""
    cache_map = {
        "IF": "optuna_cache/anomaly_thresholds_IF.pkl",
        "KF": "optuna_cache/anomaly_thresholds_KF.pkl",
        "Z-score": "optuna_cache/anomaly_thresholds_Z.pkl",
    }
    return cache_map.get(method, "optuna_cache/anomaly_thresholds_IF.pkl")  # Default to IF
