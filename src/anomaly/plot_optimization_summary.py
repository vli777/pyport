from typing import Any, List, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.logger import logger


def plot_optimization_summary(
    optimization_summary: List[Dict[str, Any]],
    max_anomaly_fraction: float,
    **plot_kwargs
) -> None:
    """
    Visualizes anomaly fractions for each stock. Only the stocks deemed anomalous
    (anomaly_fraction > max_anomaly_fraction) have their ticker label displayed above
    their anomaly fraction marker. The hover information for each point includes the
    stock ticker, threshold, and anomaly fraction.

    Args:
        optimization_summary (List[Dict[str, Any]]): List of dictionaries, each containing
            'stock', 'threshold', and 'anomaly_fraction' keys.
        max_anomaly_fraction (float): Value used to determine if a stock is anomalous.
        plot_kwargs: Additional keyword arguments for layout customization.
    """
    if not optimization_summary:
        logger.info("No optimization data to plot.")
        return

    df = pd.DataFrame(optimization_summary)
    if "stock" not in df.columns:
        logger.error("Optimization summary missing 'stock' column.")
        return

    df.sort_values("stock", inplace=True, ascending=True)

    # Identify anomalous stocks based on the provided max_anomaly_fraction.
    df["is_anomalous"] = df["anomaly_fraction"] > max_anomaly_fraction
    # Only display the ticker label for anomalous stocks.
    df["text"] = df.apply(
        lambda row: row["stock"] if row["is_anomalous"] else "", axis=1
    )

    # Create a single scatter plot for anomaly fraction.
    # Customdata holds stock, threshold, and anomaly_fraction for use in the hover template.
    custom_data = np.stack(
        (df["stock"], df["threshold"], df["anomaly_fraction"]), axis=-1
    )
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["stock"],
                y=df["anomaly_fraction"],
                mode="markers+text",
                text=df["text"],
                textposition="top center",
                customdata=custom_data,
                hovertemplate=(
                    "Asset: %{customdata[0]}<br>"
                    "Threshold: %{customdata[1]}<br>"
                    "Anomaly Fraction: %{customdata[2]}<extra></extra>"
                ),
                name="Anomaly Fraction",
            )
        ]
    )

    # Remove the ticker labels on the x-axis (middle of the chart)
    fig.update_xaxes(title_text="Asset", showticklabels=False)
    fig.update_yaxes(title_text="Anomaly Fraction")

    layout_kwargs = dict(
        title="Anomaly Fraction Summary",
        height=600,
        legend=dict(yanchor="top", y=1.05, xanchor="left", x=0),
    )
    layout_kwargs.update(plot_kwargs)
    fig.update_layout(**layout_kwargs)
    fig.show()
