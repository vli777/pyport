def filter_symbols_with_signals(
    price_df, returns_df, generate_signals_fn, mean_reversion_fn, config
):
    """
    Filter tickers using both mean reversion and weighted technical indicators.

    Args:
        price_df (pd.DataFrame): Multi-level DataFrame with columns like (TICKER, 'Open'), etc.
                                 Index = dates.
        returns_df (pd.DataFrame): Dataframe with symbol columns and returns indexed on date.
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

    # Apply mean reversion filtering if enabled
    if config.use_reversion_filter:
        # Generate mean reversion exclusions and inclusions
        mean_reversion_exclusions, mean_reversion_inclusions = mean_reversion_fn(
            returns_df=returns_df, plot=config.plot_mean_reversion
        )

        # Convert exclusions and inclusions to sets for set operations
        mean_reversion_exclusions = set(mean_reversion_exclusions)
        mean_reversion_inclusions = set(mean_reversion_inclusions)
    else:
        # Default to no exclusions or inclusions if the filter is disabled
        mean_reversion_exclusions = set()
        mean_reversion_inclusions = set()

    # Apply signal filtering if enabled
    if config.use_signal_filter:
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
        no_signal_tickers = (
            all_tickers - set(sell_signal_tickers) - set(buy_signal_tickers)
        )
    else:
        # If the filter is disabled, no signals are generated
        buy_signal_tickers = set()
        sell_signal_tickers = set()
        no_signal_tickers = set()

    # Apply mean reversion logic
    # Remove tickers in exclusions that have no signals
    filtered_set -= no_signal_tickers & mean_reversion_exclusions

    # Add tickers in inclusions that have no signals
    filtered_set |= no_signal_tickers & mean_reversion_inclusions

    # Return final filtered tickers as a list
    return list(filtered_set)
