import sys
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def create_multiday_target(returns_df, window=7, threshold=0.0):
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
    future_returns = returns_df.rolling(window=window).sum().shift(-(window - 1))
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
    # Validate category
    if category not in weighted_signals.columns.get_level_values(0):
        raise ValueError(f"Category '{category}' not found in weighted_signals.")

    # Extract category-specific columns
    category_signals = weighted_signals.loc[:, (category, slice(None))]

    if isinstance(threshold, dict):
        # Apply ticker-specific thresholds
        filtered_signals = category_signals.copy()
        for ticker, ticker_threshold in threshold.items():
            if ticker in category_signals.columns.get_level_values(1):
                filtered_signals.loc[:, (category, ticker)] = (
                    category_signals.loc[:, (category, ticker)] > ticker_threshold
                ).astype(int)
            else:
                print(f"Warning: Ticker '{ticker}' not found in category '{category}'.")
    else:
        # Apply a single threshold across all tickers
        filtered_signals = (category_signals > threshold).astype(int)

    # Remove the category level to flatten columns
    filtered_signals_flat = filtered_signals.xs(key=category, axis=1, level="Category")

    return filtered_signals_flat


def evaluate_signal_accuracy(
    category_signals: pd.DataFrame, returns_df: pd.DataFrame, threshold: float = 0.0
) -> Dict[str, float]:
    """
    Evaluate signal accuracy using precision, recall, and F1-score.
    """
    # Assign the same index name to both DataFrames if not already set
    if category_signals.index.name != "Date":
        category_signals.index.name = "Date"
        # print("Assigned 'Date' as the index name for category_signals.")

    if returns_df.index.name != "Date":
        returns_df.index.name = "Date"
        # print("Assigned 'Date' as the index name for returns_df.")

    # Ensure indices are datetime
    if not pd.api.types.is_datetime64_any_dtype(category_signals.index):
        category_signals.index = pd.to_datetime(category_signals.index)
        # print("Converted category_signals index to datetime.")

    if not pd.api.types.is_datetime64_any_dtype(returns_df.index):
        returns_df.index = pd.to_datetime(returns_df.index)
        # print("Converted returns_df index to datetime.")

    # Debugging: Print index types
    # print(f"Category Signals Index Type: {category_signals.index.dtype}")
    # print(f"Returns DF Index Type: {returns_df.index.dtype}")

    # Debugging: Check overlapping dates
    # overlapping_dates = category_signals.index.intersection(returns_df.index)
    # print(f"Number of overlapping dates: {len(overlapping_dates)}")
    # if len(overlapping_dates) == 0:
    #     print("No overlapping dates between category_signals and returns_df.")
    #     raise ValueError("No overlapping dates between category_signals and returns_df.")
    # else:
    #     print(f"Sample overlapping dates:\n{overlapping_dates[:5]}")

    # Apply threshold to create binary signals
    binary_signals = (category_signals > threshold).astype(int)

    # Flatten MultiIndex columns if needed
    if isinstance(binary_signals.columns, pd.MultiIndex):
        binary_signals.columns = binary_signals.columns.get_level_values("Ticker")

    # Create multi-day targets based on cumulative returns
    multi_day_target = create_multiday_target(
        returns_df=returns_df, window=3, threshold=0.01
    )

    # Align on dates (axis=0) and tickers (axis=1)
    binary_signals_aligned, target_aligned = binary_signals.align(
        multi_day_target, join="inner", axis=0
    )
    binary_signals_aligned, target_aligned = binary_signals_aligned.align(
        target_aligned, join="inner", axis=1
    )

    # Handle NaNs by filling with 0
    binary_signals_aligned = binary_signals_aligned.fillna(0)
    target_aligned = target_aligned.fillna(0)

    # Flatten predicted vs. actual
    predicted = binary_signals_aligned.values.flatten()
    actual = (target_aligned > 0).astype(int).values.flatten()

    # Ensure consistent lengths
    if len(predicted) != len(actual):
        raise ValueError(
            f"Predicted and actual signal lengths mismatch: {len(predicted)} vs {len(actual)}"
        )

    # Compute metrics
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def simulate_strategy_returns(
    buy_signals: pd.DataFrame,
    sell_signals: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Simulate strategy returns based on bullish and bearish signals.

    Args:
        buy_signals (pd.DataFrame): Binary buy signals (date x ticker).
        sell_signals (pd.DataFrame): Binary sell signals (date x ticker).
        returns_df (pd.DataFrame): Actual stock returns (date x ticker).

    Returns:
        dict: Cumulative returns for 'follow_all', 'avoid_bearish', and 'partial_adherence' strategies.
    """
    # 1) Align all DataFrames on date and ticker
    buy_signals_aligned, ret_aligned = buy_signals.align(
        returns_df, join="inner", axis=0
    )
    buy_signals_aligned, ret_aligned = buy_signals_aligned.align(
        ret_aligned, join="inner", axis=1
    )

    sell_signals_aligned, _ = sell_signals.align(returns_df, join="inner", axis=0)
    sell_signals_aligned, _ = sell_signals_aligned.align(
        ret_aligned, join="inner", axis=1
    )

    # 2) Strategy 1: Follow all buy signals for day t on day t
    #    Summing across tickers -> daily P/L, then cumsum
    follow_all_daily = (buy_signals_aligned * ret_aligned).sum(axis=1)
    follow_all_cumulative = follow_all_daily.cumsum()

    # 3) Strategy 2: Avoid bearish signals
    #    If sell_signals_aligned[t] == 1, do not buy that ticker on that day
    avoid_bearish_daily = (
        (1 - sell_signals_aligned) * buy_signals_aligned * ret_aligned
    ).sum(axis=1)
    avoid_bearish_cumulative = avoid_bearish_daily.cumsum()

    # 4) Strategy 3: Partial adherence (50% position if there's a bearish signal)
    partial_daily = (
        (1 - 0.5 * sell_signals_aligned) * buy_signals_aligned * ret_aligned
    ).sum(axis=1)
    partial_cumulative = partial_daily.cumsum()

    return {
        "follow_all": follow_all_cumulative,
        "avoid_bearish": avoid_bearish_cumulative,
        "partial_adherence": partial_cumulative,
    }


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
    print(f"\nAnalyzing thresholds for category: '{category}'")
    print(f"Total thresholds to evaluate: {len(thresholds)}")

    # Validate category exists in the DataFrame
    if category not in weighted_signals.columns.get_level_values(0):
        raise ValueError(f"Category '{category}' not found in weighted_signals.")

    metrics = {"F1-Score": [], "Precision": [], "Recall": []}

    # Extract category-specific signals
    category_signals = weighted_signals.loc[:, (category, slice(None))]

    for threshold in thresholds:
        try:
            # Evaluate accuracy metrics
            accuracy_metrics = evaluate_signal_accuracy(
                category_signals, returns_df, threshold=threshold
            )
            metrics["F1-Score"].append(accuracy_metrics["f1_score"])
            metrics["Precision"].append(accuracy_metrics["precision"])
            metrics["Recall"].append(accuracy_metrics["recall"])
            print(
                f"Threshold {threshold:.6f}: F1-Score={accuracy_metrics['f1_score']:.4f}, Precision={accuracy_metrics['precision']:.4f}, Recall={accuracy_metrics['recall']:.4f}"
            )
        except Exception as e:
            print(f"Error at threshold {threshold}: {e}")
            metrics["F1-Score"].append(np.nan)
            metrics["Precision"].append(np.nan)
            metrics["Recall"].append(np.nan)

    return metrics
