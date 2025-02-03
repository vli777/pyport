from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math
import tkinter

from utils.logger import logger


def plot_anomalies(
    stocks: List[str],
    returns_data: Dict[str, pd.Series],
    anomaly_flags_data: Dict[str, pd.Series],
    estimates_data: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    min_width: int = 320,  # Minimum width per subplot
    max_width: int = 1920,  # Maximum screen width
):
    """
    Plots multiple stocks' return series dynamically, ensuring each subplot
    has a minimum width of `min_width` pixels while preventing horizontal scrolling.

    Args:
        stocks (List[str]): List of stock tickers.
        returns_data (Dict[str, pd.Series]): Daily returns for each stock.
        anomaly_flags_data (Dict[str, pd.Series]): Anomaly flags (True/False) for each stock.
        estimates_data (Dict[str, np.ndarray]): Kalman estimates for each stock.
        thresholds (Dict[str, float]): Custom thresholds for each stock.
        min_width (int): Minimum width per subplot in pixels (default is 320px).
        max_width (int): Maximum figure width (default is 1920px).
    """
    total_stocks = len(stocks)
    if total_stocks == 0:
        logger.info("No anomalous stocks to plot.")
        return

    # Dynamically get screen width to prevent horizontal scrolling
    try:
        root = tkinter.Tk()
        screen_width = root.winfo_screenwidth()  # Get actual screen width
        root.destroy()
    except:
        screen_width = max_width  # Fallback if GUI unavailable

    # Determine number of columns dynamically
    available_width = min(screen_width, max_width)  # Use max available space
    cols = max(1, available_width // min_width)  # Prevent < 1 column
    cols = min(cols, total_stocks)  # Cap at total stocks
    rows = math.ceil(total_stocks / cols)  # Adjust rows accordingly

    logger.info(f"Plotting {total_stocks} stocks with {rows} rows and {cols} columns.")

    # Create a subplot grid dynamically
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=stocks,
        vertical_spacing=0.1,  # Improve readability
        horizontal_spacing=0.03,
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
        threshold = thresholds.get(stock, 1.96)

        # Returns Line
        fig.add_trace(
            go.Scatter(
                x=returns_series.index,
                y=returns_series.values,
                mode="lines",
                name=f"{stock} Returns",
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
                name=f"{stock} Kalman Estimate",
                line=dict(color="orange"),
            ),
            row=row,
            col=col,
        )

        # Highlight Anomalies
        anomalies = returns_series[anomaly_flags]

        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies.values,
                mode="markers",
                name=f"{stock} Anomaly",
                marker=dict(color="red", size=6, symbol="x"),
            ),
            row=row,
            col=col,
        )

        # Confidence Interval
        residuals = returns_series - estimates
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
                name=f"{stock} Confidence Interval",
            ),
            row=row,
            col=col,
        )

        # Update subplot labels
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Returns", row=row, col=col)

    # Set figure dimensions dynamically to prevent horizontal scrolling
    fig_width = available_width  # Uses available screen width
    fig_height = min(400 * rows, 1400)  # Limits height for readability

    fig.update_layout(
        autosize=True,
        width=fig_width,
        height=fig_height,
        title_text="Anomaly Detection Across Stocks",
        showlegend=True,
    )

    fig.show(renderer="browser")  # Opens in browser for best visibility
