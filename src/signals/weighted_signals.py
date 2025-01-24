from typing import Dict, Optional
import numpy as np
import pandas as pd

from signals.signal_categories import bullish_signals, bearish_signals


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
    combined = pd.concat(signals, axis=1)

    if weight_decay == "linear":
        w = np.linspace(1, 0.5, days)
    elif weight_decay == "exponential":
        w = np.exp(-np.linspace(0, 1, days))
    else:
        w = np.ones(days)
    w /= w.max()

    categories = ["bullish", "bearish"]
    tickers = combined.columns.levels[1]
    final_weighted = pd.DataFrame(
        0.0,  # Use 0.0 for float dtype
        index=combined.index,
        columns=pd.MultiIndex.from_product(
            [categories, tickers], names=["Category", "Ticker"]
        ),
    )

    for sig_name, wgt in signal_weights.items():
        signal_df = combined[sig_name]
        rolling_sum = pd.DataFrame(0, index=signal_df.index, columns=signal_df.columns)

        for i in range(days):
            rolling_sum += signal_df.shift(i) * w[i]

        rolling_sum *= wgt

        if sig_name in bullish_signals:
            rolling_sum.index = rolling_sum.index.astype(final_weighted.index.dtype)
            final_weighted.loc[:, ("bullish", slice(None))] += rolling_sum
        elif sig_name in bearish_signals:
            rolling_sum.index = rolling_sum.index.astype(final_weighted.index.dtype)
            final_weighted.loc[:, ("bearish", slice(None))] += rolling_sum

    return final_weighted
