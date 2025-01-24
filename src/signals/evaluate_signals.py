from typing import Dict, List, Union
import numpy as np
import pandas as pd
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


def process_multiindex_signals(
    weighted_signals: pd.DataFrame,
    category: str,
    threshold: Union[float, Dict[str, float]],
) -> pd.DataFrame:
    """
    Filters signals by category and applies threshold.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame (date x [Category, Ticker]).
        category (str): 'bullish' or 'bearish'.
        threshold (float or dict): Single float threshold or a dict with thresholds per ticker.

    Returns:
        pd.DataFrame: Binary signals (date x ticker) after threshold filtering.
    """
    category_signals = weighted_signals.loc[:, (category, slice(None))]

    if isinstance(threshold, dict):
        filtered_signals = category_signals.copy()
        for ticker, ticker_threshold in threshold.items():
            if ticker in category_signals.columns:
                filtered_signals[ticker] = (
                    category_signals[ticker] > ticker_threshold
                ).astype(int)
    else:
        filtered_signals = (category_signals > threshold).astype(int)

    return filtered_signals


def evaluate_signal_accuracy(
    weighted_signals: pd.DataFrame,
    returns_df: pd.DataFrame,
    category: str,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluates precision, recall, and F1-score for a given category and threshold.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame (date x [Category, Ticker]).
        returns_df (pd.DataFrame): Actual stock returns (date x ticker).
        category (str): 'bullish' or 'bearish'.
        threshold (float): Threshold for binary classification.

    Returns:
        dict: Metrics {"precision", "recall", "f1_score"}.
    """
    category_signals = weighted_signals.loc[:, (category, slice(None))]
    ws_aligned, ret_aligned = category_signals.align(returns_df, join="inner", axis=0)

    predicted = (ws_aligned > threshold).astype(int).values.flatten()
    actual = (ret_aligned > 0).astype(int).values.flatten()

    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def analyze_thresholds(
    weighted_signals: pd.DataFrame,
    returns_df: pd.DataFrame,
    thresholds: List[float],
    category: str,
) -> Dict[str, List[float]]:
    """
    Analyze performance metrics for various thresholds for a specific category.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame (date x [Category, Ticker]).
        returns_df (pd.DataFrame): Actual stock returns (date x ticker).
        thresholds (list): Threshold values to evaluate.
        category (str): 'bullish' or 'bearish'.

    Returns:
        dict: Metrics {"F1-Score", "Precision", "Recall"} for each threshold.
    """
    metrics = {"F1-Score": [], "Precision": [], "Recall": []}
    category_signals = weighted_signals.loc[:, (category, slice(None))]

    for threshold in thresholds:
        try:
            accuracy_metrics = evaluate_signal_accuracy(
                category_signals, returns_df, category, threshold
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


def simulate_strategy_returns(
    weighted_signals: pd.DataFrame,
    returns_df: pd.DataFrame,
    buy_threshold: float = 0.0,
    sell_threshold: float = 0.0,
) -> Dict[str, pd.Series]:
    """
    Simulates strategy returns based on thresholds for bullish and bearish signals.

    Args:
        weighted_signals (pd.DataFrame): MultiIndex DataFrame (date x [Category, Ticker]).
        returns_df (pd.DataFrame): Actual stock returns (date x ticker).
        buy_threshold (float): Threshold for bullish signals.
        sell_threshold (float): Threshold for bearish signals.

    Returns:
        dict: Strategies {"follow_all", "avoid_bearish", "partial_adherence"} with cumulative returns.
    """
    bullish_signals = weighted_signals.loc[:, ("bullish", slice(None))]
    bearish_signals = weighted_signals.loc[:, ("bearish", slice(None))]

    buy_signals = (bullish_signals > buy_threshold).astype(int)
    sell_signals = (bearish_signals > sell_threshold).astype(int)

    buy_signals, ret_aligned = buy_signals.align(returns_df, join="inner", axis=1)
    sell_signals, _ = sell_signals.align(returns_df, join="inner", axis=1)

    follow_all_daily = (buy_signals * ret_aligned).sum(axis=1)
    avoid_bearish_daily = ((1 - sell_signals) * buy_signals * ret_aligned).sum(axis=1)
    partial_daily = ((1 - 0.5 * sell_signals) * buy_signals * ret_aligned).sum(axis=1)

    return {
        "follow_all": follow_all_daily.cumsum(),
        "avoid_bearish": avoid_bearish_daily.cumsum(),
        "partial_adherence": partial_daily.cumsum(),
    }
