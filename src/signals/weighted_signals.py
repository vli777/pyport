import numpy as np
import pandas as pd

from signals.signal_categories import bullish_signals, bearish_signals

def calculate_weighted_signals(signals, signal_weights, days=7, weight_decay=None):
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
        columns=pd.MultiIndex.from_product([categories, tickers], names=["Category", "Ticker"])
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
