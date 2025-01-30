from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math

from utils import logger


def plot_anomalies(
    stocks: List[str],
    returns_data: Dict[str, pd.Series],
    anomaly_flags_data: Dict[str, pd.Series],
    estimates_data: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    stocks_per_page: int = 36,
):
    """
    Plots multiple stocks' return series in paginated grids and highlights anomalies using Plotly.

    Args:
        stocks (List[str]): List of stock names.
        returns_data (Dict[str, pd.Series]): Dictionary of daily returns for each stock, keyed by stock name.
        anomaly_flags_data (Dict[str, pd.Series]): Dictionary of anomaly flags (pd.Series) for each stock, keyed by stock name.
        estimates_data (Dict[str, np.ndarray]): Dictionary of Kalman estimates for each stock, keyed by stock name.
        thresholds (Dict[str, float]): Dictionary of thresholds for each stock, keyed by stock name.
        stocks_per_page (int): Maximum number of stocks to display per page (default is 36).
    """
    total_stocks = len(stocks)
    num_pages = math.ceil(total_stocks / stocks_per_page)

    for page in range(num_pages):
        start_idx = page * stocks_per_page
        end_idx = min(start_idx + stocks_per_page, total_stocks)
        stocks_to_plot = stocks[start_idx:end_idx]

        rows = 6
        cols = 6
        subplot_titles = stocks_to_plot
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
        )

        for i, stock in enumerate(stocks_to_plot):
            row = (i // cols) + 1
            col = (i % cols) + 1

            if stock not in returns_data:
                logger.warning(f"{stock} not found in returns_data.")
                continue

            returns_series = returns_data[stock]
            anomaly_flags = anomaly_flags_data[stock]
            estimates = estimates_data[stock]
            threshold = thresholds[stock]

            # Create trace for returns
            fig.add_trace(
                go.Scatter(
                    x=returns_series.index,
                    y=returns_series.values,
                    mode="lines",
                    name="Returns",
                    line=dict(color="blue"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Create trace for Kalman estimates
            fig.add_trace(
                go.Scatter(
                    x=returns_series.index,
                    y=estimates,
                    mode="lines",
                    name="Kalman Estimate",
                    line=dict(color="orange"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Highlight anomalies
            anomalies = returns_series[anomaly_flags]
            fig.add_trace(
                go.Scatter(
                    x=anomalies.index,
                    y=anomalies.values,
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color="red", size=6, symbol="x"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add confidence interval as filled area
            residuals = returns_series.values - estimates
            std_dev = residuals.std()
            lower_bound = estimates - 1.96 * std_dev
            upper_bound = estimates + 1.96 * std_dev

            fig.add_trace(
                go.Scatter(
                    x=returns_series.index.tolist()
                    + returns_series.index[::-1].tolist(),
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill="toself",
                    fillcolor="rgba(128, 128, 128, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="95% Confidence Interval",
                ),
                row=row,
                col=col,
            )

            # Update subplot layout
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Returns", row=row, col=col)

        fig.update_layout(
            height=3000,
            width=3000,
            title_text=f"Anomalies Page {page + 1} of {num_pages}",
            showlegend=False,
        )

        fig.show(renderer="browser")  # Opens in the default web browser
