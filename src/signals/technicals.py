import pandas as pd
import pandas_ta as ta


def calculate_stochastic_full(
    multi_df,
    window=14,
    smooth_window=3,
    d_window=3,
    mamode="ema",
    offset=0,
    fillna=0,
):
    """
    Calculate %K and %D for all tickers in a multi-level DataFrame.

    Parameters:
    - multi_df (pd.DataFrame): MultiIndex DataFrame with levels ['Ticker', 'PriceType'] where PriceType includes 'High', 'Low', 'Adj Close'.
    - window (int): The number of periods for the %K calculation. Default is 14.
    - smooth_window (int): The smoothing window for %K. Default is 3.
    - d_window (int): The window for the %D calculation. Default is 3.
    - mamode (str): Moving average mode ('sma', 'ema', etc.). Default is 'ema'.
    - offset (int): Number of periods to offset the result. Default is 0.
    - fillna (scalar or None): Value to replace NaNs. Default is 0.

    Returns:
    - stoch_k (pd.DataFrame): DataFrame containing the %K values for each ticker.
    - stoch_d (pd.DataFrame): DataFrame containing the %D values for each ticker.
    """
    # Validate MultiIndex
    if not isinstance(multi_df.columns, pd.MultiIndex):
        raise ValueError(
            "multi_df must have a MultiIndex for columns with levels [Ticker, PriceType]."
        )

    # Define required price types
    required_price_types = {"High", "Low", "Close"}
    actual_price_types = set(multi_df.columns.get_level_values(1))

    # Check if all required price types are present
    missing = required_price_types - actual_price_types
    if missing:
        raise ValueError(f"multi_df is missing required price types: {missing}")

    # Extract the list of tickers
    tickers = multi_df.columns.get_level_values(0).unique()

    # Initialize empty DataFrames for %K and %D
    stoch_k = pd.DataFrame(index=multi_df.index)
    stoch_d = pd.DataFrame(index=multi_df.index)

    # Iterate over each ticker to compute %K and %D
    for ticker in tickers:
        try:
            # Extract High, Low, and Close for the ticker
            high = multi_df[ticker]["High"]
            low = multi_df[ticker]["Low"]
            close = multi_df[ticker]["Close"]

            # Compute Stochastic Oscillator using pandas_ta
            stoch = ta.stoch(
                high=high,
                low=low,
                close=close,
                k=window,
                d=d_window,
                smooth_k=smooth_window,
                mamode=mamode,
                offset=offset,
                fillna=fillna,
            )

            # Dynamically retrieve column names from the result
            stoch_k_col, stoch_d_col = stoch.columns

            # Assign to return df
            stoch_k[ticker] = stoch[stoch_k_col]
            stoch_d[ticker] = stoch[stoch_d_col]

        except KeyError as e:
            print(f"KeyError for ticker {ticker}: {e}")
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")

    return stoch_k, stoch_d


def calculate_rainbow_stoch(
    price_df,
    windows=(5, 31, 5),
    smooth_window=3,
    d_window=3,
    mamode="ema",
    offset=0,
    fillna=0,
):
    """
    Calculate %K for a range of windows and combine the results.

    Args:
        price_df (pd.DataFrame): MultiIndex DataFrame with levels ['Ticker', 'PriceType'].
        windows (tuple): Range of windows for %K calculation.
        smooth_window, d_window, mamode, offset, fillna: Parameters for the stochastic calculation.

    Returns:
        dict: {window_size: stoch_k} for each window.
    """
    stoch_k_results = {}
    for window in range(*windows):
        stoch_k, _ = calculate_stochastic_full(
            price_df,
            window=window,
            smooth_window=smooth_window,
            d_window=d_window,
            mamode=mamode,
            offset=offset,
            fillna=fillna,
        )
        stoch_k_results[window] = stoch_k
    return stoch_k_results


def generate_convergence_signals(
    stoch_k_results, overbought=80, oversold=20, tolerance=5
):
    """
    Generate buy/sell signals based on convergence of %K across multiple windows.

    Args:
        stoch_k_results (dict): {window_size: stoch_k} for multiple windows.
        overbought (float): Overbought threshold.
        oversold (float): Oversold threshold.
        tolerance (float): Tolerance level for convergence.

    Returns:
        (pd.DataFrame, pd.DataFrame): buy_signals, sell_signals.
    """
    # Combine all %K into a single DataFrame with MultiIndex columns
    combined_stoch_k = pd.concat(
        stoch_k_results.values(),
        axis=1,
        keys=stoch_k_results.keys(),
        names=["Window", "Ticker"],
    )

    # Ensure NaN rows are ignored in calculations
    combined_stoch_k = combined_stoch_k.dropna(how="all")

    # Perform groupby operations without using axis=1
    grouped = combined_stoch_k.T.groupby(level="Ticker")
    mean_stoch_k = grouped.mean().T
    min_stoch_k = grouped.min().T
    max_stoch_k = grouped.max().T

    # Calculate convergence (difference between min and max %K across windows)
    convergence = (max_stoch_k - min_stoch_k) <= tolerance

    # Generate buy signals: convergence at oversold levels
    buy_signals = convergence & (mean_stoch_k < oversold)

    # Generate sell signals: convergence at overbought levels
    sell_signals = convergence & (mean_stoch_k > overbought)

    # Convert signals to DataFrames with tickers as columns
    buy_signals = buy_signals.fillna(0).astype(int)
    sell_signals = sell_signals.fillna(0).astype(int)

    return buy_signals, sell_signals


def calculate_macd(multi_df, fast=12, slow=26, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for each ticker
    in a multi-level DataFrame: multi_df[(ticker, 'Close')], etc.

    Returns:
        macd_line_df, macd_signal_df, macd_hist_df (pd.DataFrame each):
          Indexed by date, columns = tickers
    """
    macd_line_df = pd.DataFrame(index=multi_df.index)
    macd_signal_df = pd.DataFrame(index=multi_df.index)
    macd_hist_df = pd.DataFrame(index=multi_df.index)

    # Identify tickers (outer level)
    tickers = multi_df.columns.levels[0]

    for ticker in tickers:
        # Extract Close
        try:
            close = multi_df[(ticker, "Adj Close")]
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
    Generate MACD crossover signals for each ticker and date using vectorized operations.

    Args:
        macd_line_df (pd.DataFrame): MACD Line (columns=tickers)
        macd_signal_df (pd.DataFrame): MACD Signal Line (columns=tickers)

    Returns:
        (pd.DataFrame, pd.DataFrame):
            bullish_df, bearish_df -> Boolean DataFrames (columns=tickers)
    """
    # Current day bullish crossover: MACD line > Signal line
    bullish_current = macd_line_df > macd_signal_df

    # Previous day bullish crossover: MACD line <= Signal line
    bullish_previous = macd_line_df.shift(1) <= macd_signal_df.shift(1)

    # Current day bearish crossover: MACD line < Signal line
    bearish_current = macd_line_df < macd_signal_df

    # Previous day bearish crossover: MACD line >= Signal line
    bearish_previous = macd_line_df.shift(1) >= macd_signal_df.shift(1)

    # Bullish crossovers occur when bullish_current is True and bullish_previous is False
    bullish_df = (bullish_current & bullish_previous.shift(0).fillna(False)).fillna(
        False
    )

    # Alternatively, to directly compute crossovers:
    bullish_df = (bullish_current & (~bullish_previous)).fillna(False)
    bearish_df = (bearish_current & (~bearish_previous)).fillna(False)

    return bullish_df.astype(int), bearish_df.astype(int)


def generate_macd_preemptive_signals(
    macd_line_df, smoothing_window=5, roc_threshold=0.01, plateau_period=3
):
    """
    Generate preemptive bullish and bearish signals based on the plateauing of the MACD.

    Args:
        macd_line_df (pd.DataFrame): DataFrame containing MACD line values for each ticker.
        smoothing_window (int): Span for Exponential Weighted Mean smoothing.
        roc_threshold (float): Threshold for Rate of Change to detect plateau.
        plateau_period (int): Number of consecutive periods to confirm plateau.

    Returns:
        pd.DataFrame, pd.DataFrame: preemptive_bullish, preemptive_bearish
            Each DataFrame contains 1s (signal) and 0s (no signal) for each ticker.
    """
    # Smooth all MACD lines using EWM
    smoothed = macd_line_df.ewm(span=smoothing_window, adjust=False).mean()

    # Calculate ROC for all tickers
    roc = smoothed.pct_change()

    # Detect plateau: ROC's absolute value below threshold
    plateau = roc.abs() < roc_threshold

    # Identify plateau periods using rolling window
    plateau_signal = plateau.rolling(window=plateau_period).sum() == plateau_period

    # Fill NaNs with False
    plateau_signal = plateau_signal.fillna(False)

    # Determine direction of MACD: rising or falling
    direction = macd_line_df.diff().fillna(0) > 0  # True if rising, False if falling

    # Generate preemptive bullish and bearish signals
    preemptive_bullish = (plateau_signal & direction).astype(int)
    preemptive_bearish = (plateau_signal & ~direction).astype(int)

    return preemptive_bullish, preemptive_bearish


def calculate_adx(multi_df, length=14):
    """
    Compute ADX for all tickers in a multi-level DataFrame using pandas_ta.
    Returns a DataFrame (columns=tickers) of ADX values.

    Parameters:
    - multi_df: pd.DataFrame with MultiIndex columns where level=1 includes 'High', 'Low', 'Close'.
    - length: int, the window length for ADX calculation (default is 14).
    """
    # Validate MultiIndex columns
    if not isinstance(multi_df.columns, pd.MultiIndex):
        raise ValueError(
            "multi_df must have a MultiIndex for columns with levels including 'High', 'Low', 'Close'."
        )

    required_levels = {"High", "Low", "Close"}
    if not required_levels.issubset(set(multi_df.columns.get_level_values(1))):
        raise ValueError(
            f"multi_df columns must include the following at level 1: {required_levels}"
        )

    # Extract unique tickers
    tickers = multi_df.columns.get_level_values(0).unique()

    # Initialize a dictionary to store ADX results
    adx_results = {}

    for ticker in tickers:
        try:
            # Extract High, Low, Close for the current ticker
            high = multi_df[ticker]["High"]
            low = multi_df[ticker]["Low"]
            close = multi_df[ticker]["Close"]

            # Ensure there are enough data points
            if len(multi_df[ticker]) < length + 1:
                print(
                    f"Not enough data to compute ADX for ticker {ticker}. Required: {length + 1}, Available: {len(multi_df[ticker])}"
                )
                continue

            # Handle missing data by forward filling
            high = high.ffill().dropna()
            low = low.ffill().dropna()
            close = close.ffill().dropna()

            # Compute ADX using pandas_ta
            adx = ta.adx(high=high, low=low, close=close, length=length)

            # The ADX column is named 'ADX_{length}', e.g., 'ADX_14'
            adx_column = f"ADX_{length}"

            if adx_column in adx.columns:
                adx_results[ticker] = adx[adx_column]
            else:
                print(f"ADX column '{adx_column}' not found for ticker {ticker}.")
        except Exception as e:
            print(f"Error computing ADX for ticker {ticker}: {e}")

    # Combine all ADX results into a single DataFrame
    adx_df = pd.DataFrame(adx_results)

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


def calculate_stoch_buy_persistence(stoch_buy_signal, x=None):
    """
    Calculate stoch_buy_persistence as a binary indicator.

    Args:
        stoch_buy_signal (pd.DataFrame): DataFrame of stochastic %K values for tickers.
        x (int): Minimum consecutive days for persistence (default: empirical calculation).

    Returns:
        pd.DataFrame: Binary DataFrame indicating persistent oversold conditions.
    """
    # Determine default x if not provided
    if x is None:
        avg_persistence = (
            (stoch_buy_signal < 20)
            .astype(int)
            .apply(lambda col: col.groupby((col == 0).cumsum()).sum().mean(), axis=0)
            .mean()
        )
        x = int(round(avg_persistence))

    # Calculate persistence column-wise
    def compute_persistence(col):
        rolling_persistence = (col < 20).astype(int).rolling(
            window=x, min_periods=x
        ).sum() == x
        return rolling_persistence.astype(int)

    # Apply persistence calculation to each column
    stoch_buy_persistence = stoch_buy_signal.apply(compute_persistence, axis=0)

    return stoch_buy_persistence
