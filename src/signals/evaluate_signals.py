import numpy as np
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


def process_multiindex_signals(weighted_signals, category, threshold):
    """
    Processes MultiIndex signals to apply threshold filtering by category.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame with levels ['Category', 'Ticker'].
        category (str): 'bullish' or 'bearish'.
        threshold (float): Threshold to filter signals.

    Returns:
        pd.DataFrame: Filtered signals for the specified category.
    """
    # Filter by category
    category_signals = weighted_signals.loc[:, (category, slice(None))]

    # Apply threshold
    filtered_signals = (category_signals > threshold).astype(int)

    return filtered_signals


def evaluate_signal_accuracy(weighted_signals, returns_df, category, threshold=0.0):
    """
    Evaluate signal accuracy using precision, recall, and F1-score for a given category.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex weighted signals.
        returns_df (pd.DataFrame): Actual stock returns.
        category (str): 'bullish' or 'bearish'.
        threshold (float): Signal threshold for classification.

    Returns:
        dict: {"precision": float, "recall": float, "f1_score": float}
    """
    # Filter signals by category
    filtered_signals = process_multiindex_signals(weighted_signals, category, threshold)

    # Align shapes
    ws_aligned, ret_aligned = filtered_signals.align(returns_df, join="inner", axis=0)

    # Flatten for metric calculation
    predicted = ws_aligned.values.flatten()
    actual = (ret_aligned > 0).astype(int).values.flatten()

    # Compute metrics
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def analyze_thresholds(weighted_signals, returns_df, thresholds, category):
    """
    Analyze thresholds for a specific category ('bullish' or 'bearish').

    Args:
        weighted_signals (pd.DataFrame): MultiIndex weighted signals.
        returns_df (pd.DataFrame): Actual stock returns.
        thresholds (list or np.array): List of thresholds to analyze.
        category (str): 'bullish' or 'bearish'.

    Returns:
        dict: Metrics for each threshold.
    """
    metrics = {"F1-Score": [], "Precision": [], "Recall": []}

    for threshold in thresholds:
        try:
            accuracy_metrics = evaluate_signal_accuracy(
                weighted_signals, returns_df, category, threshold
            )
            metrics["F1-Score"].append(accuracy_metrics["f1_score"])
            metrics["Precision"].append(accuracy_metrics["precision"])
            metrics["Recall"].append(accuracy_metrics["recall"])
        except Exception as e:
            print(f"Error at threshold {threshold}: {e}")
            metrics["F1-Score"].append(np.nan)
            metrics["Precision"].append(np.nan)
            metrics["Recall"].append(np.nan)

    return metrics


def simulate_strategy_returns(weighted_signals, returns_df, threshold=0.0):
    """
    Simulate strategy returns for bullish and bearish signals.
    Args:
        weighted_signals (pd.DataFrame): Weighted signals (date x (category, ticker)).
        returns_df (pd.DataFrame): Actual returns (date x ticker).
        threshold (float): Threshold for binary classification.

    Returns:
        dict: Cumulative returns for follow_all, ignore_all, partial_adherence.
    """
    # Align DataFrames
    ws_aligned, ret_aligned = weighted_signals.align(returns_df, join="inner", axis=0)

    # Flatten MultiIndex columns to Ticker level
    ws_aligned.columns = ws_aligned.columns.get_level_values("Ticker")

    # Apply threshold to create binary signals
    predicted_binary = (ws_aligned > threshold).astype(int)

    # Align binary predictions and returns
    predicted_binary, ret_aligned = predicted_binary.align(ret_aligned, join="inner", axis=1)

    # Calculate strategy returns
    follow_all_daily = (predicted_binary * ret_aligned).sum(axis=1)
    ignore_all_daily = ((1 - predicted_binary) * ret_aligned).sum(axis=1)
    partial_daily = (
        0.5 * predicted_binary * ret_aligned + 0.5 * (1 - predicted_binary) * ret_aligned
    ).sum(axis=1)

    # Cumulative returns
    return {
        "follow_all": follow_all_daily.cumsum(),
        "ignore_all": ignore_all_daily.cumsum(),
        "partial_adherence": partial_daily.cumsum(),
    }
