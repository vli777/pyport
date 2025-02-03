from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math
import tkinter

from utils.logger import logger


def plot_anomaly_overview(
    cache: Dict[str, Any],
    returns_df: pd.DataFrame,
    min_width: int = 320,
    max_width: int = 1920,
) -> None:
    """
    Plots anomalies for stocks flagged as anomalous.

    Args:
        cache (Dict[str, Any]): Dictionary of ticker information including thresholds,
            anomaly flags, and estimates.
        returns_df (pd.DataFrame): DataFrame with stock return series.
        min_width (int): Minimum subplot width in pixels.
        max_width (int): Maximum overall width.
    """
    stocks = list(cache.keys())
    if not stocks:
        logger.info("No stocks available for plotting anomalies.")
        return

    # Determine available screen width.
    try:
        root = tkinter.Tk()
        screen_width = root.winfo_screenwidth()
        root.destroy()
    except Exception:
        screen_width = max_width

    available_width = min(screen_width, max_width)
    cols = max(1, available_width // min_width)
    cols = min(cols, len(stocks))
    rows = math.ceil(len(stocks) / cols)
    logger.info(
        f"Plotting anomalies for {len(stocks)} stocks in a grid of {rows} rows and {cols} columns."
    )

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=stocks,
        vertical_spacing=0.1,
        horizontal_spacing=0.03,
    )

    for idx, stock in enumerate(stocks):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        if stock not in returns_df.columns:
            logger.warning(f"Returns data for {stock} not found.")
            continue

        series = returns_df[stock]
        ticker_info = cache[stock]
        anomaly_flags = ticker_info["anomaly_flags"]
        estimates = ticker_info["estimates"]
        threshold = ticker_info.get("threshold", 1.96)

        # Plot returns.
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{stock} Returns",
                line=dict(color="blue"),
            ),
            row=row,
            col=col,
        )

        # Plot estimates (if available).
        if estimates is not None:
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=estimates,
                    mode="lines",
                    name=f"{stock} Estimate",
                    line=dict(color="orange"),
                ),
                row=row,
                col=col,
            )

        # Plot anomalies.
        anomalies = series[anomaly_flags]
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

        # Plot confidence interval if estimates are available.
        if estimates is not None:
            residuals = series - estimates
            std_dev = residuals.std()
            lower_bound = estimates - threshold * std_dev
            upper_bound = estimates + threshold * std_dev
            # Create a filled area between the upper and lower bounds.
            fig.add_trace(
                go.Scatter(
                    x=list(series.index) + list(series.index[::-1]),
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

        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Returns", row=row, col=col)

    fig_width = available_width
    fig_height = min(400 * rows, 1400)
    fig.update_layout(
        autosize=True,
        width=fig_width,
        height=fig_height,
        title_text="Anomaly Detection Across Stocks",
        showlegend=True,
    )
    fig.show(renderer="browser")
