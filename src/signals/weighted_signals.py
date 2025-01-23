import sys
import numpy as np
import pandas as pd

from signals.technicals import (
    calculate_macd,
    calculate_rainbow_stoch,
    generate_macd_crossovers,
    generate_macd_preemptive_signals,
    calculate_adx,
    generate_adx_signals,
    generate_convergence_signals,
)
from utils.filter import filter_signals_by_threshold


def assign_signal_weights(
    stoch_buy,
    stoch_sell,
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
    Assign weights to various signals to compute buy and sell weights.

    Args:
        stoch_buy, stoch_sell (pd.Series): Weighted stochastic buy/sell signals.
        bullish_crossover, bearish_crossover, preemptive_bullish, preemptive_bearish, adx_support (pd.Series): Other signals.
        stochastic_weight, crossover_weight, preemptive_weight, adx_weight (float): Weights for each signal.

    Returns:
        float: Buy weight.
        float: Sell weight.
    """
    # Calculate weighted buy and sell signals
    buy_weight = (
        stochastic_weight * stoch_buy
        + crossover_weight * bullish_crossover
        + preemptive_weight * preemptive_bullish
        + adx_weight * adx_support
    ).sum()

    sell_weight = (
        stochastic_weight * stoch_sell
        + crossover_weight * bearish_crossover
        + preemptive_weight * preemptive_bearish
    ).sum()

    return buy_weight, sell_weight


def calculate_weighted_signals_with_decay(
    signals, signal_weights, days=7, weight_decay="linear"
):
    """
    Calculate weighted signals with user-defined starting weights and decay.

    Args:
        signals (dict): A dictionary where keys are signal names and values are DataFrames
                        with tickers as columns and dates as index.
        signal_weights (dict): User-defined weights for each signal type.
        days (int): Number of trading days to track for signals.
        weight_decay (str): Type of weight decay ('linear' or other).

    Returns:
        pd.DataFrame: DataFrame with weighted buy/sell signals per ticker.
    """
    # Define weight decay (linear in this case)
    if weight_decay == "linear":
        decay_weights = np.linspace(
            1, 0.14, num=days
        )  # Decay from 1.0 to 0.14 over 7 days
    else:
        raise ValueError(f"Unsupported weight_decay type: {weight_decay}")

    weighted_signals = []

    for signal_name, signal_df in signals.items():
        # Extract the last 'days' rows
        signal_last_days = signal_df.iloc[-days:]

        # Apply user-defined weight and decay
        max_weight = signal_weights.get(signal_name, 0)
        daily_weights = decay_weights * max_weight
        weighted_signal = (signal_last_days * daily_weights[:, None]).sum(axis=0)

        # Store results as a DataFrame for later aggregation
        weighted_signal.name = signal_name
        weighted_signals.append(weighted_signal)

    # Combine all weighted signals into a single DataFrame
    return pd.concat(weighted_signals, axis=1)


def generate_signals(price_df):
    """
    Generate all technical signals and return a consolidated DataFrame.
    Only uses the latest date's signals without aggregation.

    Returns:
        consolidated_signals (pd.DataFrame): DataFrame containing all signal weights and indicator values per ticker.
    """
    # Generate MACD signals
    macd_line_df, macd_signal_df, macd_hist_df = calculate_macd(price_df)
    bullish_crossover, bearish_crossover = generate_macd_crossovers(
        macd_line_df, macd_signal_df
    )
    preemptive_bullish, preemptive_bearish = generate_macd_preemptive_signals(
        macd_line_df
    )

    # Calculate ADX and generate ADX trend signals
    adx_df = calculate_adx(price_df)
    adx_signals = generate_adx_signals(adx_df)

    # Calculate Stochastic Oscillator and generate convergence signals
    stoch_ks = calculate_rainbow_stoch(price_df)
    stoch_buy_signal, stoch_sell_signal = generate_convergence_signals(stoch_ks)

    # Input signal DataFrames
    signals = {
        "stoch_buy": stoch_buy_signal,
        "stoch_sell": stoch_sell_signal,
        "bullish_crossover": bullish_crossover,
        "bearish_crossover": bearish_crossover,
        "preemptive_bullish": preemptive_bullish,
        "preemptive_bearish": preemptive_bearish,
        "adx_support": adx_signals,
    }

    # User-defined weights
    signal_weights = {
        "stoch_buy": 1.0,
        "stoch_sell": 1.0,
        "bullish_crossover": 0.5,
        "bearish_crossover": 0.5,
        "preemptive_bullish": 0.3,
        "preemptive_bearish": 0.3,
        "adx_support": 0.2,
    }

    # Calculate weighted signals with decay
    weighted_signals = calculate_weighted_signals_with_decay(signals, signal_weights)
    print(weighted_signals)

    # Filter signals
    buy_signal_names = [
        "stoch_buy",
        "bullish_crossover",
        "preemptive_bullish",
        "adx_support",
    ]
    sell_signal_names = ["stoch_sell", "bearish_crossover", "preemptive_bearish"]
    buy_tickers, sell_tickers = filter_signals_by_threshold(
        weighted_signals, buy_signal_names, sell_signal_names, threshold=0.72
    )

    return buy_tickers, sell_tickers
