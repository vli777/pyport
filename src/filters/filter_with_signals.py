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
        price_df, returns_df, plot=config.plot_signal_threshold
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
    print(weighted_signals.tail())
    sys.exit()
    # Extract bullish and bearish signals
    bullish_signals = weighted_signals.loc[:, ("bullish", slice(None))]
    bearish_signals = weighted_signals.loc[:, ("bearish", slice(None))]

    # Identify buy tickers: max value of bullish signals > buy_threshold
    buy_signal_tickers = bullish_signals.max(axis=0)
    buy_signal_tickers = (
        buy_signal_tickers[buy_signal_tickers > buy_threshold]
        .get_level_values("Ticker")
        .tolist()
    )

    # Identify sell tickers: max value of bearish signals > sell_threshold
    sell_signal_tickers = bearish_signals.max(axis=0)
    sell_signal_tickers = (
        sell_signal_tickers[sell_signal_tickers > sell_threshold]
        .get_level_values("Ticker")
        .tolist()
    )

    return buy_signal_tickers, sell_signal_tickers
