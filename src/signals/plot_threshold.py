import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_threshold_metrics(
    thresholds, metrics, buy_threshold=None, sell_threshold=None
):
    """
    Plots F1-score, Precision, and Recall vs. Threshold using Seaborn.

    Args:
        thresholds (list or np.array): Threshold values.
        metrics (dict): Dictionary containing lists of metric values.
                        Keys: ["F1-Score", "Precision", "Recall"].
        buy_threshold (float, optional): Threshold for buy signals (plotted as vertical line).
        sell_threshold (float, optional): Threshold for sell signals (plotted as vertical line).
    """
    # Create a DataFrame for thresholds and metrics
    results_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "F1-Score": metrics.get("F1-Score", []),
            "Precision": metrics.get("Precision", []),
            "Recall": metrics.get("Recall", []),
        }
    )
    print(results_df.head(), results_df.shape)
    # Convert to long-form for Seaborn
    results_long = results_df.melt(
        id_vars=["threshold"],
        value_vars=["F1-Score", "Precision", "Recall"],
        var_name="Metric",
        value_name="Value",
    )

    # Plot with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_long, x="threshold", y="Value", hue="Metric", marker="o")

    # Add vertical lines for thresholds
    if buy_threshold is not None:
        plt.axvline(buy_threshold, color="g", linestyle="--", label="Buy Threshold")
    if sell_threshold is not None:
        plt.axvline(sell_threshold, color="r", linestyle="--", label="Sell Threshold")

    # Add titles and labels
    plt.title("Performance Metrics vs. Threshold", fontsize=14)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend(title="Metric", fontsize=10)
    plt.tight_layout()
    plt.show()
