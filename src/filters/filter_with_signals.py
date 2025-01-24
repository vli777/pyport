import sys
from typing import List, Tuple

import pandas as pd


def filter_symbols_with_signals(
    price_df, returns_df, generate_signals_fn, mean_reversion_fn, config
):
    """
    Filter tickers using both mean reversion and weighted technical indicators.

    Args:
        price_df (pd.DataFrame): Multi-level DataFrame with columns like (TICKER, 'Open'), etc.
                                 Index = dates.
        returns_df (pd.DataFrame): Dataframe with symbol columns and returns indexed on date
        generate_signals_fn (callable): Function to generate technical signals (returns
                                        a DataFrame with shape = date x (signal_name, ticker)).
        mean_reversion_fn (callable): Function to apply mean reversion (returns two lists of ticker strings).
        config (object): Configuration object containing plot settings and thresholds.

    Returns:
        list[str]: Final list of filtered tickers.
    """
    # Identify the set of all tickers from the price_df (outer level)
    all_tickers = set(price_df.columns.levels[0])  # Explicitly convert to a set
    filtered_set = set(all_tickers)

    # Generate mean reversion exclusions and inclusions
    mean_reversion_exclusions, mean_reversion_inclusions = mean_reversion_fn(
        price_df=price_df,
        plot=config.plot_mean_reversion,
    )

    # Convert exclusions and inclusions to sets for set operations
    mean_reversion_exclusions = set(mean_reversion_exclusions)
    mean_reversion_inclusions = set(mean_reversion_inclusions)

    # Generate signals DataFrame (date x (signal_name, ticker))
    buy_signal_tickers, sell_signal_tickers = generate_signals_fn(
        price_df,
        returns_df,
        plot=config.plot_signal_threshold,
        buy_threshold=config.buy_signal_threshold,
        sell_threshold=config.sell_signal_threshold,
    )

    # Remove tickers with sell signals
    filtered_set -= set(sell_signal_tickers)

    # Add tickers with buy signals
    filtered_set |= set(buy_signal_tickers)

    # Tickers without buy or sell signals
    no_signal_tickers = all_tickers - set(sell_signal_tickers) - set(buy_signal_tickers)

    # Apply mean reversion logic
    # Remove tickers in exclusions that have no signals
    filtered_set -= no_signal_tickers & mean_reversion_exclusions

    # Add tickers in inclusions that have no signals
    filtered_set |= no_signal_tickers & mean_reversion_inclusions

    # Return final filtered tickers as a list
    return list(filtered_set)


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
        print("Assigned 'Ticker' as the index name for sell_signal_tickers_series.")

    sell_signal_tickers = (
        sell_signal_tickers_series[sell_signal_tickers_series > sell_threshold]
        .index.get_level_values("Ticker")
        .tolist()
    )

    print(f"Buy Signal Tickers (Threshold > {buy_threshold}): {buy_signal_tickers}")
    print(f"Sell Signal Tickers (Threshold > {sell_threshold}): {sell_signal_tickers}")

    return buy_signal_tickers, sell_signal_tickers
