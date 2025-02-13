import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.logger import logger


def plot_reversion_params(data_dict):
    """
    Converts a dictionary of mean reversion parameters into a DataFrame
    and generates an interactive Plotly figure with two subplots:
      - Left: Daily Window vs. Z-Threshold (Positive and Negative)
      - Right: Weekly Window vs. Z-Threshold (Positive and Negative)

    Expects data_dict to contain the following keys (at minimum):
      - "cluster"
      - "window_daily"
      - "z_threshold_daily_positive"
      - "z_threshold_daily_negative"
      - "window_weekly"
      - "z_threshold_weekly_positive"
      - "z_threshold_weekly_negative"
    """
    if not data_dict:
        print("No data available for plotting reversion parameters.")
        return

    # Convert dictionary to DataFrame and use assets as clusters
    df = (
        pd.DataFrame.from_dict(data_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "cluster"})  # Treat asset as cluster
    )

    required_columns = {
        "window_daily",
        "z_threshold_daily_positive",
        "z_threshold_daily_negative",
        "window_weekly",
        "z_threshold_weekly_positive",
        "z_threshold_weekly_negative",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing columns in data: {missing_columns}")
        return

    # Convert all cluster values (assets) to strings and then map to numeric codes.
    df["cluster_str"] = df["cluster"].astype(str)
    unique_clusters = {
        label: idx for idx, label in enumerate(sorted(df["cluster_str"].unique()))
    }
    df["cluster_numeric"] = df["cluster_str"].map(unique_clusters)

    # Apply jitter to prevent overlap
    jitter_scale = 0.2
    df["window_daily_jitter"] = df["window_daily"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )
    df["z_threshold_daily_positive_jitter"] = df[
        "z_threshold_daily_positive"
    ] + np.random.normal(0, jitter_scale, df.shape[0])
    df["z_threshold_daily_negative_jitter"] = df[
        "z_threshold_daily_negative"
    ] + np.random.normal(0, jitter_scale, df.shape[0])
    df["window_weekly_jitter"] = df["window_weekly"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )
    df["z_threshold_weekly_positive_jitter"] = df[
        "z_threshold_weekly_positive"
    ] + np.random.normal(0, jitter_scale, df.shape[0])
    df["z_threshold_weekly_negative_jitter"] = df[
        "z_threshold_weekly_negative"
    ] + np.random.normal(0, jitter_scale, df.shape[0])

    # Create the figure with two subplots
    fig = go.Figure()

    # Left subplot: Daily parameters
    fig.add_trace(
        go.Scatter(
            x=df["window_daily_jitter"],
            y=df["z_threshold_daily_positive_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="circle",
            ),
            text=df.apply(
                lambda row: f"Asset: {row['cluster']}<br>Window (Daily): {row['window_daily']}<br>Positive Z-Threshold: {row['z_threshold_daily_positive']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Daily Positive",
            xaxis="x1",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["window_daily_jitter"],
            y=df["z_threshold_daily_negative_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="square",
            ),
            text=df.apply(
                lambda row: f"Asset: {row['cluster']}<br>Window (Daily): {row['window_daily']}<br>Negative Z-Threshold: {row['z_threshold_daily_negative']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Daily Negative",
            xaxis="x1",
            yaxis="y1",
        )
    )

    # Right subplot: Weekly parameters
    fig.add_trace(
        go.Scatter(
            x=df["window_weekly_jitter"],
            y=df["z_threshold_weekly_positive_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="circle",
            ),
            text=df.apply(
                lambda row: f"Ticker: {row['cluster']}<br>Window (Weekly): {row['window_weekly']}<br>Positive Z-Threshold: {row['z_threshold_weekly_positive']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Weekly Positive",
            xaxis="x2",
            yaxis="y2",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["window_weekly_jitter"],
            y=df["z_threshold_weekly_negative_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="square",
            ),
            text=df.apply(
                lambda row: f"Ticker: {row['cluster']}<br>Window (Weekly): {row['window_weekly']}<br>Negative Z-Threshold: {row['z_threshold_weekly_negative']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Weekly Negative",
            xaxis="x2",
            yaxis="y2",
        )
    )

    # Update layout
    fig.update_layout(
        title="Mean Reversion Clusters: Daily vs. Weekly Parameters",
        grid=dict(rows=1, columns=2, pattern="independent"),
        xaxis=dict(title="Window (Daily)", domain=[0.0, 0.45]),
        yaxis=dict(title="Z-Threshold (Daily)"),
        xaxis2=dict(title="Window (Weekly)", domain=[0.55, 1.0]),
        yaxis2=dict(title="Z-Threshold (Weekly)", anchor="x2"),
        template="plotly_white",
        legend_title="Parameter Type",
    )

    fig.show()


def plot_reversion_signals(data):
    """
    Plots Z-score based mean reversion signals using Plotly:
      - Positive => Bullish (yellow to green gradient; more positive => deeper green).
      - Negative => Bearish (orange to red gradient; more negative => deeper red).

    Args:
        data (dict): Keys are asset symbols, values are Z-score-based signals.
                     Negative => Overbought region, Positive => Oversold region.
    """
    if not data:
        print("No reversion signals available for plotting.")
        return

    # Convert data to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Asset", "Value"])
    df = df[df["Value"].apply(lambda x: isinstance(x, (int, float)))]
    if df.empty:
        print(
            "No non-zero numeric values remaining in reversion signals. Skipping plot."
        )
        return

    df = df.sort_values(by="Value")

    # Determine min and max for scaling
    min_val = df["Value"].min()  # most negative
    max_val = df["Value"].max()  # most positive
    max_abs = max(abs(min_val), abs(max_val))

    # Color gradients for negative (orange to deep red) and positive (yellow to green)
    def get_bar_color(value):
        if value < 0:
            norm = abs(value) / abs(min_val) if min_val != 0 else 0
            r = 255 - int((255 - 150) * norm)
            g = 150 - int(150 * norm)
            b = 0
        else:
            norm = value / max_val if max_val != 0 else 0
            r = 255 - int(255 * norm)
            g = 255 - int((255 - 200) * norm)
            b = 100 - int(100 * norm)
        return (r, g, b)

    bar_colors = []
    text_colors = []
    for val in df["Value"]:
        r, g, b = get_bar_color(val)
        bar_colors.append(f"rgb({r},{g},{b})")
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_colors.append("black" if brightness > 140 else "white")

    df["BarColor"] = bar_colors
    df["TextColor"] = text_colors

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
            width=0.8,
        )
    )

    fig.update_yaxes(showticklabels=True, showgrid=False, zeroline=True, showline=True)
    fig.update_xaxes(
        range=[-max_abs, max_abs],
        zeroline=True,
        zerolinecolor="grey",
        zerolinewidth=2,
        showgrid=False,
        showline=True,
        showticklabels=True,
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Arial", size=14),
        title="Z-Score Reversion Signals",
    )

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
