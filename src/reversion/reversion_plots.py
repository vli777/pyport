import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_group_reversion_params(
    group_parameters: dict, title: str = "Group Reversion Parameters"
):
    """
    Plot group-level optimized parameters using bar charts.

    For each cluster group, we plot two bar charts (daily and weekly):
      - x-axis: Rolling window (window_daily or window_weekly)
      - y-axis: Z-Threshold (z_threshold_daily or z_threshold_weekly)

    The bars are colored using a bright pastel sequential palette.
    Hover info shows the group (cluster id), the rolling window, the z-threshold,
    and a list of tickers in that group.

    Args:
        group_parameters (dict): Dictionary keyed by the original cluster label, with values like:
            {
              label: {
                 "tickers": [list of tickers],
                 "params": {
                     "window_daily": ...,
                     "z_threshold_daily": ...,
                     "window_weekly": ...,
                     "z_threshold_weekly": ...,
                     "weight_daily": ...,
                     "weight_weekly": ...,
                     "cluster": <group_id as hash string>
                 }
              },
              ...
            }
        title (str): Overall title for the plot.

    Returns:
        go.Figure: The resulting Plotly figure.
    """
    # Build a summary DataFrame from group_parameters.
    records = []
    for label, grp_data in group_parameters.items():
        params = grp_data["params"]
        tickers = grp_data["tickers"]
        records.append(
            {
                "group_label": label,
                "cluster": params.get(
                    "cluster", label
                ),  # use the cluster id from params if available
                "window_daily": params.get("window_daily", 20),
                "z_threshold_daily": params.get("z_threshold_daily", 1.5),
                "window_weekly": params.get("window_weekly", 5),
                "z_threshold_weekly": params.get("z_threshold_weekly", 1.5),
                "tickers": ", ".join(tickers),
            }
        )
    df = pd.DataFrame(records)

    # Choose a pastel sequential palette. For example, using Plotly Express "Viridis" but
    # we can also choose a palette with bright pastel colors.
    # Here, we use a simple categorical mapping based on the number of groups.
    num_groups = df.shape[0]
    colors = px.colors.qualitative.Vivid_r
    # If there are more groups than colors, cycle through.
    df["color"] = [colors[i % len(colors)] for i in range(num_groups)]

    # Create subplots: one for daily parameters, one for weekly parameters.
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Daily Parameters", "Weekly Parameters")
    )

    # Daily bar chart: each bar represents a group.
    fig.add_trace(
        go.Bar(
            x=df["window_daily"],
            y=df["cluster"],
            orientation="h",
            marker=dict(color=df["color"], line=dict(width=0)),
            text=df["z_threshold_daily"],
            hovertemplate=(
                "Cluster: %{y}<br>"
                + "Window: %{x}<br>"
                + "Z-Threshold: %{text}<br>"
                + "Tickers: "
                + df["tickers"]
                + "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Weekly bar chart.
    fig.add_trace(
        go.Bar(
            x=df["window_weekly"],
            y=df["cluster"],
            orientation="h",
            marker=dict(color=df["color"], line=dict(width=0)),
            text=df["z_threshold_weekly"],
            hovertemplate=(
                "Cluster: %{y}<br>"
                + "Window: %{x}<br>"
                + "Z-Threshold: %{text}<br>"
                + "Tickers: "
                + df["tickers"]
                + "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # Update axes titles.
    fig.update_xaxes(title_text="Rolling Window (Days)", row=1, col=1)
    fig.update_xaxes(title_text="Rolling Window (Weeks)", row=1, col=2)
    fig.update_yaxes(title_text="Cluster ID", automargin=True)

    fig.update_layout(
        title=title, margin=dict(l=100, r=50, t=100, b=50), showlegend=False
    )

    fig.show()
    return fig


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
        width=1200,
        height=800,
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
