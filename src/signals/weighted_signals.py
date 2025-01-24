import numpy as np
import pandas as pd


def calculate_weighted_signals(signals, signal_weights, days=7, weight_decay=None):
    """
    Returns a time-series DataFrame (date x ticker) where each row's signal
    is a decayed sum of the previous `days` rows from each signal.
    """
    # 1) Combine signals into a single DataFrame with MultiIndex columns
    #    columns = (signal_name, ticker)
    combined = pd.concat(signals, axis=1)

    # 2) Prepare decay weights
    #    Example: linear or exponential
    if weight_decay == "linear":
        w = np.linspace(1, 0.5, days)
    elif weight_decay == "exponential":
        w = np.exp(-np.linspace(0, 1, days))
    else:
        w = np.ones(days)
    # Normalize so max weight = 1.0
    w /= w.max()

    # 3) For each signal_name, we want a rolling sum of the last `days` rows * w
    #    Then multiply by the signal_weight for that signal_name.
    #    We'll accumulate all signals into one final output with shape = (dates, tickers).

    # Create an empty df with the same date index & ticker columns for final output
    final_weighted = pd.DataFrame(
        0, index=combined.index, columns=combined.columns.levels[1]
    )

    for sig_name, wgt in signal_weights.items():
        # Extract only the columns for this signal (one col per ticker)
        signal_df = combined[sig_name]

        # Create a rolling sum with custom weights using shift & multiply,
        # then sum across 'days' windows
        # Example: for i in [0..days-1], do signal_df.shift(i)*w[i], sum across i
        rolling_sum = pd.DataFrame(0, index=signal_df.index, columns=signal_df.columns)
        for i in range(days):
            rolling_sum += signal_df.shift(i) * w[i]

        # Multiply decayed rolling sum by weight
        rolling_sum *= wgt

        # Accumulate into final
        final_weighted += rolling_sum

    # final_weighted is date x ticker
    # final_weighted = final_weighted / final_weighted.max(axis=0)

    return final_weighted
