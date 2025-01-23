def filter_symbols_with_signals(
    price_df, generate_signals_fn, mean_reversion_fn, config
):
    """
    Filter tickers using both mean reversion and weighted technical indicators.

    Args:
        price_df (pd.DataFrame): Multi-level DataFrame with columns like (TICKER, 'Open'), etc.
                                 Index = dates.
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
    buy_signal_tickers, sell_signal_tickers = generate_signals_fn(price_df)

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
    weighted_signals, buy_signal_names, sell_signal_names, threshold=0.72
):
    """
    Filter tickers with aggregated buy/sell weights exceeding a threshold.

    Args:
        weighted_signals (pd.DataFrame): DataFrame with weighted buy/sell signals.
        buy_signal_names (list): List of signal names contributing to buy weight.
        sell_signal_names (list): List of signal names contributing to sell weight.
        threshold (float): Threshold for filtering signals.

    Returns:
        pd.Index: Tickers exceeding the buy/sell threshold.
    """
    # Aggregate buy and sell weights
    weighted_signals["Buy_Signal_Weight"] = weighted_signals[buy_signal_names].sum(
        axis=1
    )
    weighted_signals["Sell_Signal_Weight"] = weighted_signals[sell_signal_names].sum(
        axis=1
    )

    # Filter tickers by threshold
    buy_signal_tickers = weighted_signals[
        weighted_signals["Buy_Signal_Weight"] > threshold
    ].index
    sell_signal_tickers = weighted_signals[
        weighted_signals["Sell_Signal_Weight"] > threshold
    ].index

    return buy_signal_tickers, sell_signal_tickers
