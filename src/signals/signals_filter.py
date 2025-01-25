import sys
from typing import List, Tuple

import pandas as pd


def filter_signals_by_threshold(
    weighted_signals: pd.DataFrame,
    buy_threshold: float = 1.0,
    sell_threshold: float = 1.0,
) -> Tuple[List[str], List[str]]:
    """
    Filters buy and sell signals based on thresholds, separately for bullish and bearish categories.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame (date x [Category, Ticker]).
        buy_threshold (float): Threshold to classify buy signals for bullish signals.
        sell_threshold (float): Threshold to classify sell signals for bearish signals.

    Returns:
        tuple: (buy_signal_tickers, sell_signal_tickers)
    """
    # Verify that weighted_signals has the expected MultiIndex
    if not isinstance(weighted_signals.columns, pd.MultiIndex):
        raise TypeError(
            "weighted_signals must have a MultiIndex for columns with levels ['Category', 'Ticker']."
        )

    expected_categories = ["bullish", "bearish"]
    actual_categories = (
        weighted_signals.columns.get_level_values("Category").unique().tolist()
    )
    missing_categories = set(expected_categories) - set(actual_categories)
    if missing_categories:
        raise ValueError(
            f"Missing categories in weighted_signals: {missing_categories}"
        )

    # Extract bullish and bearish signals
    try:
        bullish_signals = weighted_signals.loc[:, ("bullish", slice(None))]
        bearish_signals = weighted_signals.loc[:, ("bearish", slice(None))]
    except KeyError as e:
        raise KeyError(f"Error extracting categories: {e}")

    # Check for any NaNs after verification
    if bullish_signals.isnull().values.any() or bearish_signals.isnull().values.any():
        raise ValueError(
            "NaN values detected in bullish or bearish signals after verification."
        )

    # Identify buy tickers: max value of bullish signals > buy_threshold
    buy_signal_tickers_series = bullish_signals.max(axis=0)

    # Ensure the Series has the 'Ticker' level name
    if buy_signal_tickers_series.index.name != "Ticker":
        buy_signal_tickers_series.index.name = "Ticker"
        # print("Assigned 'Ticker' as the index name for buy_signal_tickers_series.")

    buy_signal_tickers = (
        buy_signal_tickers_series[buy_signal_tickers_series > buy_threshold]
        .index.get_level_values("Ticker")
        .tolist()
    )

    # Identify sell tickers: max value of bearish signals > sell_threshold
    sell_signal_tickers_series = bearish_signals.max(axis=0)

    # Ensure the Series has the 'Ticker' level name
    if sell_signal_tickers_series.index.name != "Ticker":
        sell_signal_tickers_series.index.name = "Ticker"
        # print("Assigned 'Ticker' as the index name for sell_signal_tickers_series.")

    sell_signal_tickers = (
        sell_signal_tickers_series[sell_signal_tickers_series > sell_threshold]
        .index.get_level_values("Ticker")
        .tolist()
    )

    print(f"Buy Signal Tickers (Threshold > {buy_threshold}): {buy_signal_tickers}")
    print(f"Sell Signal Tickers (Threshold > {sell_threshold}): {sell_signal_tickers}")

    return buy_signal_tickers, sell_signal_tickers
