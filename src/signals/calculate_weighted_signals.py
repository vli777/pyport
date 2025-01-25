import sys
from typing import Dict, Optional
import numpy as np
import pandas as pd
import json


def calculate_weighted_signals(
    signals: Dict[str, pd.DataFrame],
    signal_weights: Dict[str, float],
    days: int = 7,
    weight_decay: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate weighted signals for bullish and bearish categories.

    Args:
        signals (Dict[str, pd.DataFrame]): A dictionary where keys are signal names and
            values are DataFrames (date x ticker) with binary (0/1) signals.
        signal_weights (Dict[str, float]): A dictionary of weights for each signal.
            Keys must match the keys in the `signals` argument.
        days (int, optional): Number of days to apply the decay over. Default is 7.
        weight_decay (Optional[str], optional): Type of weight decay to apply:
            - "linear": Linearly decaying weights from 1.0 to 0.5.
            - "exponential": Exponentially decaying weights.
            - None: No decay (all weights are equal). Default is None.

    Returns:
        pd.DataFrame: A MultiIndex DataFrame with levels ['Category', 'Ticker'],
            where:
            - 'Category' contains "bullish" and "bearish".
            - 'Ticker' contains tickers from the input DataFrames.
            Each value represents the weighted signal strength for the given date,
            ticker, and category.
    """
    # Combine all signals into a single MultiIndex DataFrame
    combined = pd.concat(signals, axis=1)

    # Define decay weights
    if weight_decay == "linear":
        decay_weights = np.linspace(1, 0.5, days)
    elif weight_decay == "exponential":
        decay_weights = np.exp(-np.linspace(0, 1, days))
    else:
        decay_weights = np.ones(days)
    decay_weights /= decay_weights.sum()  # Normalize decay weights

    # Define categories
    categories = ["bullish", "bearish"]
    tickers = combined.columns.get_level_values(1).unique()
    final_weighted = pd.DataFrame(
        0.0,  # Initialize with zeros
        index=combined.index,
        columns=pd.MultiIndex.from_product(
            [categories, tickers], names=["Category", "Ticker"]
        ),
    )

    # Process each signal
    for sig_name, wgt in signal_weights.items():
        if sig_name not in signals:
            print(f"Warning: Signal '{sig_name}' not found in signals.")
            continue

        signal_df = signals[sig_name]
        rolling_sum = pd.DataFrame(
            0.0, index=signal_df.index, columns=signal_df.columns
        )

        # Apply rolling weighted sum
        for i in range(days):
            shifted = signal_df.shift(i).fillna(0)  # Shift signal for past days
            rolling_sum += shifted * decay_weights[i]

        # Apply signal weight
        rolling_sum *= wgt

        with open("signal_categories.json", "r") as f:
            data = json.load(f)
            bullish_signals = data["bullish_signals"]
            bearish_signals = data["bearish_signals"]

        # Assign to bullish or bearish category
        if sig_name in bullish_signals:
            for ticker in signal_df.columns:
                final_weighted.loc[:, ("bullish", ticker)] += rolling_sum[ticker]
        elif sig_name in bearish_signals:
            for ticker in signal_df.columns:
                final_weighted.loc[:, ("bearish", ticker)] += rolling_sum[ticker]
        else:
            print(
                f"Warning: Signal '{sig_name}' not classified as 'bullish' or 'bearish'."
            )

    # Replace NaNs with zeros
    final_weighted.fillna(0, inplace=True)

    return final_weighted


def verify_weighted_signals(weighted_signals: pd.DataFrame):
    """
    Verifies that the weighted_signals DataFrame is correctly populated and free of NaNs.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame with ['Category', 'Ticker'] levels.

    Raises:
        ValueError: If any NaNs are found in the DataFrame.
    """
    if weighted_signals.isnull().values.any():
        print("Error: 'weighted_signals' contains NaN values.")
        print(weighted_signals.isnull().sum())
        raise ValueError(
            "NaN values detected in 'weighted_signals'. Please check signal processing steps."
        )
    else:
        print("Verification Passed: 'weighted_signals' is free of NaNs.")


def verify_ticker_consistency(weighted_signals: pd.DataFrame, returns_df: pd.DataFrame):
    """
    Verifies that tickers in weighted_signals match those in returns_df.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame (date x [Category, Ticker]).
        returns_df (pd.DataFrame): Actual stock returns (date x ticker).

    Raises:
        ValueError: If there are mismatched tickers.
    """
    tickers_weighted = set(weighted_signals.columns.get_level_values("Ticker"))
    tickers_returns = set(returns_df.columns)

    missing_in_returns = tickers_weighted - tickers_returns
    missing_in_weighted = tickers_returns - tickers_weighted

    if missing_in_returns:
        raise ValueError(
            f"Tickers present in weighted_signals but missing in returns_df: {missing_in_returns}"
        )
    if missing_in_weighted:
        print(
            f"Warning: Tickers present in returns_df but missing in weighted_signals: {missing_in_weighted}"
        )
    else:
        print(
            "Ticker consistency verified: All tickers in weighted_signals are present in returns_df."
        )
