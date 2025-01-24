from sklearn.metrics import precision_score, recall_score, f1_score


def create_multiday_target(returns_df, window=3, threshold=0.0):
    """
    Create a multi-day target for signal evaluation.

    Args:
        returns_df (pd.DataFrame): Actual returns DataFrame (index=date, columns=tickers).
        window (int): Number of days for cumulative return evaluation.
        threshold (float): Return threshold for buy/sell classification.

    Returns:
        pd.DataFrame: Binary target (1 for buy, 0 for no buy),
                      shape = (dates, tickers).
    """
    # Calculate rolling cumulative returns over the next `window` days
    future_returns = returns_df.shift(-window + 1).rolling(window=window).sum()
    target = (future_returns > threshold).astype(int)
    return target


def evaluate_signal_accuracy(weighted_signals, returns_df, threshold=0.0):
    """
    Evaluate signal accuracy using precision, recall, and F1-score.

    Args:
        weighted_signals (pd.DataFrame): Weighted buy/sell signals.
        returns_df (pd.DataFrame): Actual stock returns.
        threshold (float): Signal threshold for classification.

    Returns:
        dict: {"precision": float, "recall": float, "f1_score": float}
    """
    # 1) Align shapes
    ws_aligned, ret_aligned = weighted_signals.align(returns_df, join="inner", axis=0)

    # 2) Flatten predicted vs. actual
    predicted = (ws_aligned > threshold).astype(int).values.flatten()
    actual = (ret_aligned > 0).astype(int).values.flatten()

    # 3) Compute metrics
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def simulate_strategy_returns(weighted_signals, returns_df, threshold=0.0):
    """
    Simulate a few simple strategies for demonstration:
      - follow_all: if signals > threshold, hold that ticker
      - ignore_all: always do the opposite
      - partial_adherence: 50-50

    Args:
        weighted_signals (pd.DataFrame): Weighted signals (dates x tickers).
        returns_df (pd.DataFrame): Actual returns (dates x tickers).
        threshold (float): threshold for deciding "in" or "out" of a ticker.

    Returns:
        dict of pd.Series: Each key is a strategy, each value is the cumulative returns Series.
    """
    # 1) Align shapes
    ws_aligned, ret_aligned = weighted_signals.align(returns_df, join="inner", axis=0)

    # 2) Predicted binary signals
    predicted_binary = (ws_aligned > threshold).astype(int)

    # 3) Strategy returns
    follow_all_daily = (predicted_binary * ret_aligned).sum(axis=1)
    follow_all_cumulative = follow_all_daily.cumsum()

    ignore_all_daily = ((1 - predicted_binary) * ret_aligned).sum(axis=1)
    ignore_all_cumulative = ignore_all_daily.cumsum()

    partial_daily = (
        0.5 * predicted_binary * ret_aligned
        + 0.5 * (1 - predicted_binary) * ret_aligned
    ).sum(axis=1)
    partial_cumulative = partial_daily.cumsum()

    return {
        "follow_all": follow_all_cumulative,
        "ignore_all": ignore_all_cumulative,
        "partial_adherence": partial_cumulative,
    }
