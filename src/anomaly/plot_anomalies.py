from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math

from utils.logger import logger


def plot_anomalies(
    stocks: List[str],
    returns_data: Dict[str, pd.Series],
    anomaly_flags_data: Dict[str, pd.Series],
    estimates_data: Dict[str, np.ndarray],
    thresholds: Dict[str, float],  # Now actively used
    cols: int = 6,
):
    """
    Plots multiple stocks' return series in a single scrollable Plotly figure
    and highlights anomalies.

    Args:
        stocks (List[str]): List of stock names.
        returns_data (Dict[str, pd.Series]): Dictionary of daily returns for each stock.
        anomaly_flags_data (Dict[str, pd.Series]): Dictionary of anomaly flags (pd.Series).
        estimates_data (Dict[str, np.ndarray]): Dictionary of Kalman estimates for each stock.
        thresholds (Dict[str, float]): Custom thresholds for each stock, used to refine anomaly detection.
        cols (int): Number of columns in the grid (default is 6).
    """
    total_stocks = len(stocks)
    if total_stocks == 0:
        logger.info("No anomalous stocks to plot.")
        return
    
    rows = math.ceil(total_stocks / cols)  # Adjust rows based on total stocks

    # Create a subplot figure with dynamic rows
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=stocks,
        vertical_spacing=0.08,  # Adjust spacing for readability
        horizontal_spacing=0.05,
    )

    for i, stock in enumerate(stocks):
        row = (i // cols) + 1
        col = (i % cols) + 1

        if stock not in returns_data:
            logger.warning(f"{stock} not found in returns_data.")
            continue

        returns_series = returns_data[stock]
        anomaly_flags = anomaly_flags_data[stock]
        estimates = estimates_data[stock]
        threshold = thresholds.get(stock, 1.96)  # Default to 1.96 if not specified

        # Returns Line
        fig.add_trace(
            go.Scatter(
                x=returns_series.index,
                y=returns_series.values,
                mode="lines",
                name="Returns" if i == 0 else "",
                line=dict(color="blue"),
            ),
            row=row,
            col=col,
        )

        # Kalman Estimates
        fig.add_trace(
            go.Scatter(
                x=returns_series.index,
                y=estimates,
                mode="lines",
                name="Kalman Estimate" if i == 0 else "",
                line=dict(color="orange"),
            ),
            row=row,
            col=col,
        )

        # Highlight Anomalies Above Threshold
        residuals = returns_series - estimates
        anomaly_flags = residuals.abs() > (threshold * residuals.std())  # Use threshold
        anomalies = returns_series[anomaly_flags]

        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies.values,
                mode="markers",
                name="Anomaly" if i == 0 else "",
                marker=dict(color="red", size=6, symbol="x"),
            ),
            row=row,
            col=col,
        )

        # Confidence Interval Using Threshold
        std_dev = residuals.std()
        lower_bound = estimates - threshold * std_dev
        upper_bound = estimates + threshold * std_dev

        fig.add_trace(
            go.Scatter(
                x=returns_series.index.tolist() + returns_series.index[::-1].tolist(),
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Confidence Interval" if i == 0 else "",
            ),
            row=row,
            col=col,
        )

        # Update subplot labels
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Returns", row=row, col=col)

    # Adjust figure size dynamically based on the number of rows
    fig.update_layout(
        height=400 * rows,  # Dynamically adjust height for scrolling
        width=2000,
        title_text="Anomaly Detection Across Stocks",
        showlegend=True,
    )

    fig.show(renderer="browser")  # Opens in the default web browser
