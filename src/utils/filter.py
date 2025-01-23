def filter_symbols_with_signals(
    price_df, generate_signals_fn, mean_reversion_fn, config
):
    """
    Filter tickers using both mean reversion and weighted technical indicators.

    Args:
        price_df (pd.DataFrame): DataFrame containing price data (columns are ticker symbols).
        generate_signals_fn (callable): Function to generate technical signals (e.g., generate_signals).
        mean_reversion_fn (callable): Function to apply mean reversion (e.g., apply_mean_reversion).
        config (object): Configuration object containing plot settings and thresholds.

    Returns:
        list[str]: Final list of filtered tickers.
    """
    # Apply mean reversion
    mean_reversion_exclusions, mean_reversion_inclusions = mean_reversion_fn(
        price_df=price_df,
        plot=config.plot_mean_reversion,
    )

    # Apply weighted technical indicators
    signals_df = generate_signals_fn(price_df)

    # Extract buy and sell signals
    buy_signals = signals_df["Buy_Signal_Weight"] > 0
    sell_signals = signals_df["Sell_Signal_Weight"] > 0

    # Initialize a set of tickers from the price DataFrame
    filtered_set = set(price_df.columns)

    # Update filtered symbols based on generate_signals (priority over mean reversion)
    for ticker in price_df.columns:
        if ticker in sell_signals[sell_signals].index:  # Sell signal
            filtered_set.discard(ticker)  # Exclude ticker
        elif ticker in buy_signals[buy_signals].index:  # Buy signal
            filtered_set.add(ticker)  # Include ticker
        else:
            # Apply mean reversion only if generate_signals doesn't conflict
            if ticker in mean_reversion_exclusions:
                filtered_set.discard(ticker)  # Exclude ticker
            if ticker in mean_reversion_inclusions:
                filtered_set.add(ticker)  # Include ticker

    # Return the updated list of filtered symbols
    return list(filtered_set)
