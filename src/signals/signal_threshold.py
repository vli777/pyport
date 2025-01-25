from typing import Dict, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from signals.evaluate_signal_metrics import evaluate_signal_accuracy


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


def plot_threshold_metrics(
    thresholds,
    bullish_metrics,
    bearish_metrics,
    buy_threshold=None,
    sell_threshold=None,
):
    """
    Plots F1-score, Precision, and Recall vs. Threshold for bullish and bearish signals in a single figure.

    Args:
        thresholds (list or np.array): Threshold values.
        bullish_metrics (dict): Dictionary containing metrics for bullish signals.
        bearish_metrics (dict): Dictionary containing metrics for bearish signals.
        buy_threshold (float, optional): Threshold for buy signals (plotted as vertical line in the first subplot).
        sell_threshold (float, optional): Threshold for sell signals (plotted as vertical line in the second subplot).
    """
    # Prepare data for bullish metrics
    bullish_df = (
        pd.DataFrame(
            {
                "threshold": thresholds,
                "F1-Score": bullish_metrics.get("F1-Score", []),
                "Precision": bullish_metrics.get("Precision", []),
                "Recall": bullish_metrics.get("Recall", []),
            }
        )
        .melt(
            id_vars=["threshold"],
            value_vars=["F1-Score", "Precision", "Recall"],
            var_name="Metric",
            value_name="Value",
        )
        .dropna(subset=["Value"])
    )

    # Prepare data for bearish metrics
    bearish_df = (
        pd.DataFrame(
            {
                "threshold": thresholds,
                "F1-Score": bearish_metrics.get("F1-Score", []),
                "Precision": bearish_metrics.get("Precision", []),
                "Recall": bearish_metrics.get("Recall", []),
            }
        )
        .melt(
            id_vars=["threshold"],
            value_vars=["F1-Score", "Precision", "Recall"],
            var_name="Metric",
            value_name="Value",
        )
        .dropna(subset=["Value"])
    )

    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    sns.set(style="whitegrid")

    # Plot bullish metrics
    sns.lineplot(
        data=bullish_df,
        x="threshold",
        y="Value",
        hue="Metric",
        marker="o",
        ax=axes[0],
    )
    if buy_threshold is not None:
        axes[0].axvline(buy_threshold, color="g", linestyle="--", label="Buy Threshold")
    axes[0].set_title("Bullish Signals Metrics")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Metric Value")
    axes[0].legend(title="Metric")

    # Plot bearish metrics
    sns.lineplot(
        data=bearish_df,
        x="threshold",
        y="Value",
        hue="Metric",
        marker="o",
        ax=axes[1],
    )
    if sell_threshold is not None:
        axes[1].axvline(
            sell_threshold, color="r", linestyle="--", label="Sell Threshold"
        )
    axes[1].set_title("Bearish Signals Metrics")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("")

    # Finalize layout
    plt.tight_layout()
    plt.show()
