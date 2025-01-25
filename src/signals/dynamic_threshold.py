from typing import Dict, List
import numpy as np
import pandas as pd

from signals.evaluate_signals import evaluate_signal_accuracy


def get_dynamic_thresholds(
    returns_df, window=20, overbought_multiplier=1.0, oversold_multiplier=1.0
):
    """
    Calculate ticker-specific dynamic overbought and oversold thresholds with asymmetric multipliers.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame with tickers as columns and dates as index.
        window (int): Rolling window size for Z-score calculation.
        overbought_multiplier (float): Multiplier for overbought threshold.
        oversold_multiplier (float): Multiplier for oversold threshold.

    Returns:
        dict: {ticker: (overbought_threshold, oversold_threshold)}
    """
    dynamic_thresholds = {}

    for ticker in returns_df.columns:
        # Calculate Z-scores for the given ticker
        rolling_mean = returns_df[ticker].rolling(window).mean()
        rolling_std = returns_df[ticker].rolling(window).std()
        z_scores = (returns_df[ticker] - rolling_mean) / rolling_std

        # Calculate thresholds using asymmetric multipliers
        z_std = z_scores.std()  # Standard deviation of Z-scores
        overbought_threshold = overbought_multiplier * z_std
        oversold_threshold = -oversold_multiplier * z_std

        dynamic_thresholds[ticker] = (overbought_threshold, oversold_threshold)

    return dynamic_thresholds


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
