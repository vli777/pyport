import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.logger import logger


def plot_reversion_params(data_dict):
    """
    Converts a dictionary of mean reversion parameters into a DataFrame
    and generates an interactive Plotly figure with two subplots:
    - Left: Daily Window vs. Z-Threshold
    - Right: Weekly Window vs. Z-Threshold

    Features:
    - Hovering shows Ticker, Window, Z-Threshold, and Cluster
    - Jittering applied to prevent overlapping points

    Parameters:
    data_dict (dict): Dictionary with tickers as keys and parameter dicts as values.
                      Expected format:
                      {
                          "AAPL": {"window_daily": 30, "z_threshold_daily": 1.5, "window_weekly": 5, "z_threshold_weekly": 2.0, "cluster": 1},
                          "TSLA": {"window_daily": 25, "z_threshold_daily": 1.2, "window_weekly": 6, "z_threshold_weekly": 2.5, "cluster": 2},
                          ...
                      }
    """

    # Convert dictionary to DataFrame
    df = (
        pd.DataFrame.from_dict(data_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "ticker"})
    )

    # Apply jitter to prevent overlap
    jitter_scale = 0.2
    df["window_daily_jitter"] = df["window_daily"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )
    df["z_threshold_daily_jitter"] = df["z_threshold_daily"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )
    df["window_weekly_jitter"] = df["window_weekly"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )
    df["z_threshold_weekly_jitter"] = df["z_threshold_weekly"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )

    # Create subplots
    fig = go.Figure()

    # Left subplot: Daily Window vs. Z-Threshold
    fig.add_trace(
        go.Scatter(
            x=df["window_daily_jitter"],
            y=df["z_threshold_daily_jitter"],
            mode="markers",
            marker=dict(color=df["cluster"], colorscale="viridis", size=8, opacity=0.7),
            text=df.apply(
                lambda row: f"Ticker: {row['ticker']}<br>Window: {row['window_daily']}<br>Z-Threshold: {row['z_threshold_daily']}<br>Cluster: {row['cluster']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Daily",
            xaxis="x1",
            yaxis="y1",
        )
    )

    # Right subplot: Weekly Window vs. Z-Threshold
    fig.add_trace(
        go.Scatter(
            x=df["window_weekly_jitter"],
            y=df["z_threshold_weekly_jitter"],
            mode="markers",
            marker=dict(color=df["cluster"], colorscale="viridis", size=8, opacity=0.7),
            text=df.apply(
                lambda row: f"Ticker: {row['ticker']}<br>Window: {row['window_weekly']}<br>Z-Threshold: {row['z_threshold_weekly']}<br>Cluster: {row['cluster']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Weekly",
            xaxis="x2",
            yaxis="y2",
        )
    )

    # Update layout with two subplots
    fig.update_layout(
        title="Mean Reversion Clusters: Daily vs. Weekly",
        grid=dict(rows=1, columns=2, pattern="independent"),
        xaxis=dict(title="Window (Daily)", domain=[0.0, 0.45]),
        yaxis=dict(title="Z-Threshold"),
        xaxis2=dict(title="Window (Weekly)", domain=[0.55, 1.0]),
        yaxis2=dict(title="Z-Threshold", anchor="x2"),
        template="plotly_white",
        legend_title="Dataset",
    )

    fig.show()


def plot_reversion_signals(data):
    """
    Plots Z-score based mean reversion signals using Plotly:
      - Positive => Bullish (yellow to green gradient; more positive => deeper green).
      - Negative => Bearish (orange to red gradient; more negative => deeper red).

    Args:
        data (dict): Keys are asset tickers, values are Z-score-based signals.
                     Negative => Overbought region, Positive => Oversold region.
    """

    # Convert data to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Asset", "Value"])
    df = df[df["Value"] != 0]  # Remove zero values if any
    if df.empty:
        logger.info("No non-zero values remaining in reversion signals. Skipping plot.")
        return

    df = df.sort_values(by="Value")

    # Determine min and max for scaling
    min_val = df["Value"].min()  # most negative
    max_val = df["Value"].max()  # most positive
    # For symmetric axis, we span from -max_abs to +max_abs
    max_abs = max(abs(min_val), abs(max_val))

    # Color gradients
    #   Negative: small negative (close to 0) -> orange, large negative -> deep red
    #   Positive: small positive (close to 0) -> yellow, large positive -> deep green
    def get_bar_color(value):
        if value < 0:
            # Normalize in [0, 1], 0 => near 0, 1 => min_val
            norm = abs(value) / abs(min_val) if min_val != 0 else 0
            # Orange (255,150,0) -> Red (150,0,0)
            r = 255 - int((255 - 150) * norm)  # 255 -> 150
            g = 150 - int(150 * norm)  # 150 -> 0
            b = 0
        else:
            # Normalize in [0, 1], 0 => 0, 1 => max_val
            norm = value / max_val if max_val != 0 else 0
            # Yellow (255,255,100) -> Green (0,200,0)
            r = 255 - int(255 * norm)  # 255 -> 0
            g = 255 - int((255 - 200) * norm)  # 255 -> 200
            b = 100 - int(100 * norm)  # 100 -> 0
        return (r, g, b)

    # Determine bar color and text color
    bar_colors = []
    text_colors = []
    for val in df["Value"]:
        r, g, b = get_bar_color(val)
        bar_colors.append(f"rgb({r},{g},{b})")
        # Decide if bar is dark or light for text contrast
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_colors.append("black" if brightness > 140 else "white")

    df["BarColor"] = bar_colors
    df["TextColor"] = text_colors

    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Asset"],
            x=df["Value"],
            orientation="h",
            marker=dict(color=df["BarColor"]),
            text=df["Asset"],
            textposition="inside",
            textfont=dict(color=df["TextColor"], size=16),
            hoverinfo="x+y+text",
            width=0.8,  # Thicker bars
        )
    )

    # Hide y-axis ticks/labels (tickers now appear only inside bars)
    fig.update_yaxes(
        showticklabels=True,
        showgrid=False,
        zeroline=True,
        showline=True,
    )

    # Center the x-axis around 0, keep the zero line visible
    fig.update_xaxes(
        range=[-max_abs, max_abs],
        zeroline=True,  # Show line at x=0
        zerolinecolor="grey",
        zerolinewidth=2,
        showgrid=False,
        showline=True,
        showticklabels=True,  # Hide numeric tick labels
    )

    # Increase overall figure size for easier visibility
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Arial", size=14),
        title="Z-Score Reversion Signals",
    )

    # Add left/right annotations for Overbought/Oversold
    # Rotated 90°, centered vertically
    fig.add_annotation(
        x=0.0,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Overbought</b>",
        showarrow=False,
        font=dict(size=16),
        align="center",
        textangle=-90,
        xanchor="left",
        yanchor="middle",
    )
    fig.add_annotation(
        x=1.0,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Oversold</b>",
        showarrow=False,
        font=dict(size=16),
        align="center",
        textangle=90,
        xanchor="right",
        yanchor="middle",
    )

    fig.show()
