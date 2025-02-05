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
                "window_weekly": params.get("window_weekly", 20),
                "z_threshold_weekly": params.get("z_threshold_weekly", 1.5),
                "tickers": ", ".join(tickers),
            }
        )
    df = pd.DataFrame(records)

    # Choose a pastel sequential palette. For example, using Plotly Express "Viridis" but
    # we can also choose a palette with bright pastel colors.
    # Here, we use a simple categorical mapping based on the number of groups.
    num_groups = df.shape[0]
    colors = px.colors.qualitative.Pastel1
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
