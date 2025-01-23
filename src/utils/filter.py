import pandas as pd

from utils import logger

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

    # 1) Identify the set of all tickers from the price_df (outer level)
    all_tickers = price_df.columns.levels[0]

    # 2) Apply mean reversion
    #    Typically returns something like:
    #       mean_reversion_exclusions = ['AAPL', 'GOOG', ...]
    #       mean_reversion_inclusions = ['MSFT', ...]
    mean_reversion_exclusions, mean_reversion_inclusions = mean_reversion_fn(
        price_df=price_df,
        plot=config.plot_mean_reversion,
    )

    # 3) Generate signals DataFrame (date x (signal_name, ticker)).
    #    Example columns might be: [('Buy_Signal_Weight', 'AAPL'), ('Buy_Signal_Weight', 'MSFT'), ...]
    signals_df = generate_signals_fn(price_df)

    # 4) If we only need the latest signals (the last date):
    latest_signals = signals_df.iloc[
        -1
    ]  # Series with multi-level index => (signal_name, ticker)

    # Extract each signal from the multi-index:
    # latest_signals.loc["Buy_Signal_Weight"] will be a Series indexed by tickers.
    try:
        buy_signal_series = latest_signals.loc["Buy_Signal_Weight"]
        sell_signal_series = latest_signals.loc["Sell_Signal_Weight"]
    except KeyError as e:
        logger.error(f"Missing expected signal columns: {e}")
        raise

    # 5) Convert to booleans: buy > 0, sell > 0
    buy_signal_bool = buy_signal_series > 0
    sell_signal_bool = sell_signal_series > 0

    # 6) Start with all_tickers as a set
    filtered_set = set(all_tickers)

    # 7) Update filtered symbols based on generate_signals (priority over mean reversion)
    for ticker in all_tickers:
        if sell_signal_bool.get(ticker, False):
            # If there's a sell signal for this ticker => discard it
            filtered_set.discard(ticker)
        elif buy_signal_bool.get(ticker, False):
            # If there's a buy signal => include it
            filtered_set.add(ticker)
        else:
            # If no buy/sell signal, apply mean reversion logic
            if ticker in mean_reversion_exclusions:
                filtered_set.discard(ticker)
            if ticker in mean_reversion_inclusions:
                filtered_set.add(ticker)

    # Return the updated list of filtered symbols
    return list(filtered_set)
