import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_robust_z_scores(returns_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate rolling robust z-scores using the rolling median and MAD.
    The MAD is scaled by 1.4826 to approximate the standard deviation.
    """
    rolling_median = returns_df.rolling(window=window, min_periods=1).median()
    # Rolling MAD: use a lambda to compute median absolute deviation over the window
    mad = returns_df.rolling(window=window, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    # Avoid division by zero by replacing zeros in mad
    mad.replace(0, np.nan, inplace=True)
    robust_z = (returns_df - rolling_median) / (mad * 1.4826)
    return robust_z.fillna(0)


def plot_robust_z_scores(
    robust_z: pd.DataFrame,
    z_threshold: float,
    n_cols: int = 6,
    title: str = "Robust Z-Scores",
):
    """
    Create an interactive Plotly grid of subplots for robust z-scores for multiple tickers.
    Each subplot displays the z-score time series for one ticker along with horizontal lines
    at +z_threshold (overbought) and -z_threshold (oversold).

    Args:
        robust_z (pd.DataFrame): DataFrame with dates as index and tickers as columns containing robust z-scores.
        z_threshold (float): The threshold value used for overbought/oversold signals.
                             Overbought line will be at z_threshold and oversold at -z_threshold.
        n_cols (int): Number of columns in the grid. The number of rows is computed based on the number of tickers.
        title (str): Title for the overall figure.

    Returns:
        None: Displays the Plotly figure.
    """
    tickers = robust_z.columns.tolist()
    n_tickers = len(tickers)
    n_rows = int(np.ceil(n_tickers / n_cols))

    # Create a subplot grid with a dynamic number of rows
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=tickers)

    # Iterate over each ticker and add its time series and threshold lines
    for i, ticker in enumerate(tickers):
        row = i // n_cols + 1
        col = i % n_cols + 1

        series = robust_z[ticker]
        x_values = series.index
        y_values = series.values

        # Add the robust z-score trace
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=f"{ticker} Z-Score",
                line=dict(color="blue"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Determine x-range for the threshold lines
        if len(x_values) > 0:
            start_date = x_values.min()
            end_date = x_values.max()
        else:
            start_date, end_date = None, None

        # Add horizontal overbought threshold line at +z_threshold
        fig.add_trace(
            go.Scatter(
                x=[start_date, end_date],
                y=[z_threshold, z_threshold],
                mode="lines",
                name="Overbought",
                line=dict(color="red", dash="dash"),
                showlegend=(i == 0),  # Show legend only for the first subplot
            ),
            row=row,
            col=col,
        )

        # Add horizontal oversold threshold line at -z_threshold
        fig.add_trace(
            go.Scatter(
                x=[start_date, end_date],
                y=[-z_threshold, -z_threshold],
                mode="lines",
                name="Oversold",
                line=dict(color="green", dash="dash"),
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

    # Update the layout with an appropriate height to allow scrolling if needed
    fig.update_layout(
        height=300 * n_rows,
        width=1200,
        title=title,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05),
    )
    fig.show()
