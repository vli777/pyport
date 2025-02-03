from typing import Any, List, Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.logger import logger


def plot_optimization_summary(
    optimization_summary: List[Dict[str, Any]], **plot_kwargs
) -> None:
    """
    Visualizes the optimization summary for each stock in a two-panel plot:
    one for thresholds and one for scores and anomaly fractions.

    Args:
        optimization_summary (List[Dict[str, Any]]): List of dictionaries, each containing
            'stock', 'threshold', 'best_score', and 'anomaly_fraction'.
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
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Optimal Thresholds per Stock", "Scores & Anomaly Fraction"),
        specs=[[{}], [{"secondary_y": True}]],
    )

    # Plot thresholds.
    fig.add_trace(
        go.Bar(x=df["stock"], y=df["threshold"], name="Threshold"),
        row=1,
        col=1,
    )

    # Plot best scores if available.
    if "best_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["stock"],
                y=df["best_score"],
                mode="lines+markers",
                name="Best Score",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
    else:
        logger.warning("Column 'best_score' not found in optimization summary.")

    # Plot anomaly fraction.
    if "anomaly_fraction" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["stock"],
                y=df["anomaly_fraction"],
                mode="lines+markers",
                name="Anomaly Fraction",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
    else:
        logger.warning("Column 'anomaly_fraction' not found in optimization summary.")

    fig.update_xaxes(title_text="Stock Ticker", row=2, col=1)
    fig.update_yaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Score", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Anomaly Fraction", secondary_y=True, row=2, col=1)

    layout_kwargs = dict(
        title="Optimization Summary",
        height=800,
        legend=dict(yanchor="top", y=1.05, xanchor="left", x=0),
    )
    layout_kwargs.update(plot_kwargs)
    fig.update_layout(**layout_kwargs)
    fig.show()
