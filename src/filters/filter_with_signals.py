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
    weighted_signals,
    buy_threshold=1.0,
    sell_threshold=1.0,
):
    """
    Filters buy and sell signals based on thresholds, separately for bullish and bearish categories.

    Args:
        weighted_signals (pd.DataFrame): Weighted signals with MultiIndex columns (Category, Ticker).
        buy_threshold (float): Threshold to classify buy signals for bullish signals.
        sell_threshold (float): Threshold to classify sell signals for bearish signals.

    Returns:
        tuple: (buy_signal_tickers, sell_signal_tickers)
    """
    # Separate bullish and bearish signals
    bullish_signals = weighted_signals.loc[:, "bullish"]
    bearish_signals = weighted_signals.loc[:, "bearish"]

    # Identify buy tickers: bullish signals above buy_threshold
    buy_signal_tickers = bullish_signals.columns[
        bullish_signals.max(axis=0) > buy_threshold
    ].tolist()

    # Identify sell tickers: bearish signals above sell_threshold
    sell_signal_tickers = bearish_signals.columns[
        bearish_signals.max(axis=0) > sell_threshold
    ].tolist()

    return buy_signal_tickers, sell_signal_tickers
