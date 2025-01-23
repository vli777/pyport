import pandas as pd

from signals.technicals import (
    calculate_macd,
    generate_macd_crossovers,
    generate_macd_preemptive_signals,
    calculate_adx,
    generate_adx_signals,
    calculate_stochastic_full,
    generate_convergence_signal,
)


def assign_signal_weights(
    stoch_buy_df,
    stoch_sell_df,
    macd_bull_df,
    macd_bear_df,
    macd_pre_bull_df,
    macd_pre_bear_df,
    adx_support_df,
    stoch_weight=1.0,
    crossover_weight=0.5,
    preemptive_weight=0.3,
    adx_weight=0.2,
):
    """
    Each arg is a DataFrame: index=dates, columns=tickers, with boolean signals.
    Returns two DataFrames of floats (buy/sell weights).
    """
    # Convert booleans to floats for weighting
    stoch_buy_w = stoch_buy_df.astype(float) * stoch_weight
    stoch_sell_w = stoch_sell_df.astype(float) * stoch_weight

    macd_bull_w = macd_bull_df.astype(float) * crossover_weight
    macd_bear_w = macd_bear_df.astype(float) * crossover_weight

    pre_bull_w = macd_pre_bull_df.astype(float) * preemptive_weight
    pre_bear_w = macd_pre_bear_df.astype(float) * preemptive_weight

    adx_w = adx_support_df.astype(float) * adx_weight

    buy_weight_df = stoch_buy_w + macd_bull_w + pre_bull_w + adx_w
    sell_weight_df = stoch_sell_w + macd_bear_w + pre_bear_w + adx_w

    return buy_weight_df, sell_weight_df


def generate_signals(
    price_df,
    stochastic_windows=[5, 10, 15, 20, 25, 30],
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    adx_window=14,
    adx_threshold=25,
    stochastic_smooth_k=3,
    stochastic_smooth_d=3,
    stochastic_tolerance=5,
    stochastic_overbought=80,
    stochastic_oversold=20,
    stochastic_weight=1.0,
    crossover_weight=0.5,
    preemptive_weight=0.3,
    adx_weight=0.2,
):
    """
    Generate weighted signals based on stochastic convergence, MACD (crossover and preemptive), and ADX.

    Args:
        price_df (pd.DataFrame): DataFrame containing price data of tickers.
        stochastic_windows (list): List of window intervals for Stochastic calculations.
        macd_fast (int): Fast EMA window for MACD.
        macd_slow (int): Slow EMA window for MACD.
        macd_signal (int): Signal line window for MACD.
        adx_window (int): Window for ADX calculation.
        adx_threshold (float): Threshold for ADX to consider the trend strong.
        stochastic_smooth_k (int): Smoothing period for %K.
        stochastic_smooth_d (int): Smoothing period for %D.
        stochastic_tolerance (float): Tolerance level for stochastic convergence.
        stochastic_overbought (float): Overbought threshold for stochastic signals.
        stochastic_oversold (float): Oversold threshold for stochastic signals.
        stochastic_weight (float): Weight for stochastic signals (absolute).
        crossover_weight (float): Weight for MACD crossovers.
        preemptive_weight (float): Weight for MACD preemptive signals.
        adx_weight (float): Weight for ADX signals.

    Returns:
        pd.DataFrame: Combined signal weights for buy and sell signals.
    """
    # Calculate stochastic weighted averages
    stoch_k, stoch_d = calculate_stochastic_full(
        price_df,
        windows=stochastic_windows,
        smooth_k=stochastic_smooth_k,
        smooth_d=stochastic_smooth_d,
    )

    # Generate stochastic convergence signals
    stochastic_buy, stochastic_sell = generate_convergence_signal(
        stoch_k,
        stoch_d,
        overbought=stochastic_overbought,
        oversold=stochastic_oversold,
        tolerance=stochastic_tolerance,
    )

    # Calculate MACD components
    macd_line_df, macd_signal_df, macd_hist_df = calculate_macd(
        price_df, fast=macd_fast, slow=macd_slow, signal=macd_signal
    )

    # Generate MACD crossover and preemptive signals
    bullish_crossover_df, bearish_crossover_df = generate_macd_crossovers(
        macd_line_df, macd_signal_df
    )
    preemptive_bullish_df = generate_macd_preemptive_signals(macd_line_df)
    preemptive_bearish_df = generate_macd_preemptive_signals(-macd_line_df)

    # Calculate ADX
    adx_df = calculate_adx(price_df, window=adx_window)
    # Generate ADX signals
    adx_support_df = generate_adx_signals(adx_df, adx_threshold)

    # Assign weights to all signals
    buy_signal_weight, sell_signal_weight = assign_signal_weights(
        stoch_buy_df=stochastic_buy,
        stoch_sell_df=stochastic_sell,
        macd_bull_df=bullish_crossover_df,
        macd_bear_df=bearish_crossover_df,
        macd_pre_bull_df=preemptive_bullish_df,
        macd_pre_bear_df=preemptive_bearish_df,
        adx_support_df=adx_support_df,
        stoch_weight=stochastic_weight,
        crossover_weight=crossover_weight,
        preemptive_weight=preemptive_weight,
        adx_weight=adx_weight,
    )

    # Create a multi-level DataFrame for the signals
    signals_list = []
    for label, df_sig in [
        ("Buy_Signal_Weight", buy_signal_weight),
        ("Sell_Signal_Weight", sell_signal_weight),
        ("Stochastic_Buy", stochastic_buy),
        ("Stochastic_Sell", stochastic_sell),
        ("MACD_Bullish_Crossover", bullish_crossover_df),
        ("MACD_Bearish_Crossover", bearish_crossover_df),
        ("MACD_Preemptive_Bullish", preemptive_bullish_df),
        ("MACD_Preemptive_Bearish", preemptive_bearish_df),
        ("ADX_Support", adx_support_df),
    ]:
        # Ensure the DataFrame is aligned
        df_sig = df_sig.reindex(price_df.index).reindex(
            price_df.columns.levels[0], axis=1
        )
        # Assign a MultiIndex to columns: (signal_name, ticker)
        df_sig.columns = pd.MultiIndex.from_product([[label], df_sig.columns])
        signals_list.append(df_sig)

    # Concatenate all signals horizontally
    signals = pd.concat(signals_list, axis=1)

    return signals
