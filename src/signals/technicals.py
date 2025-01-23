import pandas as pd
import pandas_ta as ta


def calculate_stochastic_full(
    price_df, windows=[5, 10, 15, 20, 25, 30], smooth_k=3, smooth_d=3
):
    """
    Calculate the weighted average of StochasticFull indicators across multiple windows
    for each ticker in a multi-level DataFrame.

    Assumes price_df has columns of the form:
      (TICKER, 'High'), (TICKER, 'Low'), (TICKER, 'Close'), etc.

    Returns:
        stoch_k_final (pd.DataFrame): Weighted %K for each ticker (columns=tickers).
        stoch_d_final (pd.DataFrame): Weighted %D for each ticker (columns=tickers).
    """
    # Prepare output DataFrames, index = same as price_df
    stoch_k_final = pd.DataFrame(index=price_df.index)
    stoch_d_final = pd.DataFrame(index=price_df.index)

    # Identify the set of tickers from the outer level
    tickers = price_df.columns.levels[0]

    for ticker in tickers:
        # Safely extract High/Low/Close columns
        # Skip this ticker if any are missing
        try:
            high = price_df[(ticker, "High")]
            low = price_df[(ticker, "Low")]
            close = price_df[(ticker, "Close")]
        except KeyError:
            # For example, if (ticker,'High') is missing
            continue

        # Weighted sums over multiple windows
        weight_sum = sum(range(1, len(windows) + 1))
        weighted_k = pd.Series(0, index=price_df.index)
        weighted_d = pd.Series(0, index=price_df.index)

        # Compute Stochastic for each window
        for i, window in enumerate(windows):
            stoch_data = ta.stoch(
                high=high,
                low=low,
                close=close,
                k=window,
                smooth_k=smooth_k,
                smooth_d=smooth_d,
                mamode="ema",
                offset=0,
            )
            # Columns from pandas_ta: e.g. "STOCHK_5_3_3", "STOCHD_5_3_3"
            col_k = f"STOCHk_{window}_{smooth_k}_{smooth_d}"
            col_d = f"STOCHd_{window}_{smooth_k}_{smooth_d}"

            weighted_k += stoch_data[col_k] * (i + 1)
            weighted_d += stoch_data[col_d] * (i + 1)

        # Compute average
        avg_k = weighted_k / weight_sum
        avg_d = weighted_d / weight_sum

        # Store in final DataFrames
        stoch_k_final[ticker] = avg_k
        stoch_d_final[ticker] = avg_d

    return stoch_k_final, stoch_d_final


def generate_convergence_signal(
    stoch_k, stoch_d, overbought=80, oversold=20, tolerance=5
):
    """
    Generate buy/sell signals for each date and ticker when %K and %D converge
    at overbought or oversold levels.

    Args:
        stoch_k (pd.DataFrame): Weighted average %K, columns=tickers
        stoch_d (pd.DataFrame): Weighted average %D, columns=tickers
        overbought (float): Overbought threshold
        oversold (float): Oversold threshold
        tolerance (float): Tolerance level for convergence

    Returns:
        (pd.DataFrame, pd.DataFrame): buy_signals, sell_signals
          Each is a boolean DF with the same shape as stoch_k/stoch_d.
    """
    diff = (stoch_k - stoch_d).abs()
    convergence = diff <= tolerance

    buy_signals = convergence & (stoch_k < oversold) & (stoch_d < oversold)
    sell_signals = convergence & (stoch_k > overbought) & (stoch_d > overbought)

    buy_signals = buy_signals.fillna(0)
    sell_signals = sell_signals.fillna(0)

    return buy_signals, sell_signals


def calculate_macd(price_df, fast=12, slow=26, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for each ticker
    in a multi-level DataFrame: price_df[(ticker, 'Close')], etc.

    Returns:
        macd_line_df, macd_signal_df, macd_hist_df (pd.DataFrame each):
          Indexed by date, columns = tickers
    """
    macd_line_df = pd.DataFrame(index=price_df.index)
    macd_signal_df = pd.DataFrame(index=price_df.index)
    macd_hist_df = pd.DataFrame(index=price_df.index)

    # Identify tickers (outer level)
    tickers = price_df.columns.levels[0]

    for ticker in tickers:
        # Extract Close
        try:
            close = price_df[(ticker, "Close")]
        except KeyError:
            continue  # This ticker may not have a 'Close' column

        # Compute MACD using pandas_ta
        macd_data = ta.macd(close, fast=fast, slow=slow, signal=signal)

        # Extract the generated columns
        col_line = f"MACD_{fast}_{slow}_{signal}"
        col_signal = f"MACDs_{fast}_{slow}_{signal}"
        col_hist = f"MACDh_{fast}_{slow}_{signal}"

        # Store in our DataFrames
        macd_line_df[ticker] = macd_data[col_line]
        macd_signal_df[ticker] = macd_data[col_signal]
        macd_hist_df[ticker] = macd_data[col_hist]

    return macd_line_df, macd_signal_df, macd_hist_df


def generate_macd_crossovers(macd_line_df, macd_signal_df):
    """
    Generate MACD crossover signals for each ticker and date.

    Args:
        macd_line_df (pd.DataFrame): MACD Line (columns=tickers)
        macd_signal_df (pd.DataFrame): MACD Signal Line (columns=tickers)

    Returns:
        (pd.DataFrame, pd.DataFrame):
            bullish_df, bearish_df -> Boolean DataFrames (columns=tickers)
    """
    bullish_df = pd.DataFrame(index=macd_line_df.index)
    bearish_df = pd.DataFrame(index=macd_line_df.index)

    for ticker in macd_line_df.columns:
        line = macd_line_df[ticker]
        signal = macd_signal_df[ticker]

        bullish = (line > signal) & (line.shift(1) <= signal.shift(1))
        bearish = (line < signal) & (line.shift(1) >= signal.shift(1))

        bullish_df[ticker] = bullish.fillna(False)
        bearish_df[ticker] = bearish.fillna(False)

    return bullish_df, bearish_df


def generate_macd_preemptive_signals(
    macd_line_df, smoothing_window=5, roc_threshold=0.01, plateau_period=3
):
    """
    Generate preemptive signals based on the plateauing of the MACD,
    returning a DataFrame of booleans for each ticker/date.
    """
    preemptive_df = pd.DataFrame(index=macd_line_df.index)

    for ticker in macd_line_df.columns:
        macd_line = macd_line_df[ticker]

        # Smooth
        smoothed = macd_line.ewm(span=smoothing_window, adjust=False).mean()
        # ROC
        roc = smoothed.pct_change()
        # Plateau detection
        plateau = roc.abs() < roc_threshold
        plateau_signal = plateau.rolling(window=plateau_period).sum() == plateau_period
        plateau_signal = plateau_signal.fillna(False)

        preemptive_df[ticker] = plateau_signal

    return preemptive_df


def calculate_adx(price_df, window=14):
    """
    Compute ADX for each ticker in a multi-level DataFrame.
    Returns a DataFrame (columns=tickers) of ADX values.
    """
    adx_df = pd.DataFrame(index=price_df.index)
    tickers = price_df.columns.levels[0]

    for ticker in tickers:
        try:
            high = price_df[(ticker, "High")]
            low = price_df[(ticker, "Low")]
            close = price_df[(ticker, "Close")]
        except KeyError:
            continue

        adx_data = ta.adx(high=high, low=low, close=close, length=window)
        # Typically yields columns: ['ADX_14', 'DMP_14', 'DMN_14']
        adx_col = f"ADX_{window}"
        adx_df[ticker] = adx_data[adx_col]

    return adx_df


def generate_adx_signals(adx_df, adx_threshold=25):
    """
    Generate ADX trend signals based on a specified threshold for all dates.

    Args:
        adx_df (pd.DataFrame): DataFrame containing ADX values with tickers as columns.
        adx_threshold (float): Threshold above which a trend is considered strong.

    Returns:
        pd.DataFrame: DataFrame with boolean ADX trend signals for each ticker across all dates.
    """
    # Generate boolean signals where ADX > threshold
    adx_trending = adx_df > adx_threshold
    adx_trending = adx_trending.fillna(False)  # Handle any remaining NaNs

    return adx_trending.astype(
        int
    )  # Convert boolean to integer (1 for True, 0 for False)
