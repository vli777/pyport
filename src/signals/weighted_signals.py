import pandas as pd

from macd import (
    calculate_macd,
    generate_macd_crossovers,
    generate_macd_preemptive_signals,
)
from adx import calculate_adx, generate_adx_signals
from stochastics import calculate_stochastic_full, generate_convergence_signal


def assign_signal_weights(
    stochastic_signal,
    bullish_crossover,
    bearish_crossover,
    preemptive_bullish,
    preemptive_bearish,
    adx_support,
    stochastic_weight=1.0,
    crossover_weight=0.5,
    preemptive_weight=0.3,
    adx_weight=0.2,
):
    """
    Assign weights to stochastic, MACD (crossover and preemptive), and ADX signals.

    Args:
        stochastic_signal (pd.Series): Boolean Series for stochastic absolute signals.
        bullish_crossover (pd.Series): Boolean Series for MACD bullish crossovers.
        bearish_crossover (pd.Series): Boolean Series for MACD bearish crossovers.
        preemptive_bullish (pd.Series): Boolean Series for preemptive bullish signals.
        preemptive_bearish (pd.Series): Boolean Series for preemptive bearish signals.
        adx_support (pd.Series): Boolean Series for ADX trend confirmation.
        stochastic_weight (float): Weight for stochastic signals (absolute).
        crossover_weight (float): Weight for MACD crossovers.
        preemptive_weight (float): Weight for MACD preemptive signals.
        adx_weight (float): Weight for ADX signals.

    Returns:
        pd.Series, pd.Series: Weighted bullish and bearish signals.
    """
    # Initialize weights
    bullish_weights = stochastic_signal.astype(float) * stochastic_weight
    bearish_weights = stochastic_signal.astype(float) * stochastic_weight

    # Add weights for definite MACD crossovers
    bullish_weights += bullish_crossover.astype(float) * crossover_weight
    bearish_weights += bearish_crossover.astype(float) * crossover_weight

    # Add weights for MACD preemptive signals
    bullish_weights += preemptive_bullish.astype(float) * preemptive_weight
    bearish_weights += preemptive_bearish.astype(float) * preemptive_weight

    # Add weights for ADX confirmation
    bullish_weights += adx_support.astype(float) * adx_weight
    bearish_weights += adx_support.astype(float) * adx_weight

    return bullish_weights, bearish_weights


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
    macd_line, macd_signal_line, macd_hist = calculate_macd(
        price_df, fast=macd_fast, slow=macd_slow, signal=macd_signal
    )

    # Generate MACD crossover and preemptive signals
    bullish_crossover, bearish_crossover = generate_macd_crossovers(
        macd_line, macd_signal_line
    )
    preemptive_bullish = generate_macd_preemptive_signals(macd_line)
    preemptive_bearish = generate_macd_preemptive_signals(-macd_line)

    # Calculate ADX
    adx = calculate_adx(price_df, window=adx_window)

    # Generate ADX signals
    adx_support = generate_adx_signals(adx, adx_threshold)

    # Assign weights to all signals
    buy_signal_weight, sell_signal_weight = assign_signal_weights(
        stochastic_signal=stochastic_buy,
        bullish_crossover=bullish_crossover,
        bearish_crossover=bearish_crossover,
        preemptive_bullish=preemptive_bullish,
        preemptive_bearish=preemptive_bearish,
        adx_support=adx_support,
        stochastic_weight=stochastic_weight,
        crossover_weight=crossover_weight,
        preemptive_weight=preemptive_weight,
        adx_weight=adx_weight,
    )

    # Create a DataFrame for the signals
    signals = pd.DataFrame(
        {
            "Buy_Signal_Weight": buy_signal_weight,
            "Sell_Signal_Weight": sell_signal_weight,
            "Stochastic_Buy": stochastic_buy,
            "Stochastic_Sell": stochastic_sell,
            "MACD_Bullish_Crossover": bullish_crossover,
            "MACD_Bearish_Crossover": bearish_crossover,
            "MACD_Preemptive_Bullish": preemptive_bullish,
            "MACD_Preemptive_Bearish": preemptive_bearish,
            "ADX_Support": adx_support,
        },
        index=price_df.index,
    )

    return signals
