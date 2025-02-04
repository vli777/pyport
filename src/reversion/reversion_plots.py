import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_reversion_bubble(ticker_params: dict, title: str = "Reversion Parameters"):
    """
    Create a two-column bubble plot:
      - Left: Daily parameters
            x-axis: window_daily (and used for marker color)
            y-axis: ticker (categorical)
            bubble size: proportional to z_threshold_daily
            hover: ticker, window_daily, z_threshold_daily, weight_daily
      - Right: Weekly parameters
            x-axis: window_weekly (and used for marker color)
            y-axis: ticker (categorical)
            bubble size: proportional to z_threshold_weekly
            hover: ticker, window_weekly, z_threshold_weekly, weight_weekly

    The marker color is based on the rolling window value using a sequential gradient ("Viridis").

    Args:
        ticker_params (dict): Dictionary keyed by ticker with parameters, e.g.
            {
                'AAPL': {
                    'window_daily': 25,
                    'z_threshold_daily': 1.7,
                    'weight_daily': 0.5,
                    'window_weekly': 30,
                    'z_threshold_weekly': 1.9,
                    'weight_weekly': 0.55
                },
                ...
            }
        title (str): Overall title for the figure.

    Returns:
        go.Figure: The resulting Plotly figure.
    """
    # Convert the dictionary to a DataFrame.
    df = pd.DataFrame.from_dict(ticker_params, orient="index")
    df.index.name = "ticker"
    df.reset_index(inplace=True)

    # Fill missing values for safety.
    df["z_threshold_daily"] = df["z_threshold_daily"].fillna(0)
    df["z_threshold_weekly"] = df["z_threshold_weekly"].fillna(0)
    df["window_daily"] = df["window_daily"].fillna(0)
    df["window_weekly"] = df["window_weekly"].fillna(0)

    # Sort by ticker for consistency.
    df = df.sort_values(by="ticker")

    # Define a scale factor for bubble size.
    size_scale = 15

    # Create a subplot with 1 row and 2 columns.
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Daily Parameters", "Weekly Parameters"),
        horizontal_spacing=0.1,
    )

    # Left subplot: Daily parameters.
    # Marker color is based on window_daily using a sequential color scale.
    fig.add_trace(
        go.Scatter(
            x=df["window_daily"],
            y=df["ticker"],
            mode="markers",
            marker=dict(
                size=df["z_threshold_daily"] * size_scale,
                color=df["window_daily"],
                colorscale="Viridis",
                colorbar=dict(title="Daily Window"),
                opacity=0.8,
                line=dict(width=1, color="black"),
            ),
            hovertemplate=(
                "Ticker: %{text}<br>"
                + "Window: %{x}<br>"
                + "Z-Threshold: %{customdata[0]}<br>"
            ),
            text=df["ticker"],
            customdata=df[["z_threshold_daily"]].values,
        ),
        row=1,
        col=1,
    )

    # Right subplot: Weekly parameters.
    # Marker color is based on window_weekly using the same sequential color scale.
    fig.add_trace(
        go.Scatter(
            x=df["window_weekly"],
            y=df["ticker"],
            mode="markers",
            marker=dict(
                size=df["z_threshold_weekly"] * size_scale,
                color=df["window_weekly"],
                colorscale="Viridis",
                colorbar=dict(title="Weekly Window"),
                opacity=0.8,
                line=dict(width=1, color="black"),
            ),
            hovertemplate=(
                "Ticker: %{text}<br>"
                + "Window: %{x}<br>"
                + "Z-Threshold: %{customdata[0]}<br>"
            ),
            text=df["ticker"],
            customdata=df[["z_threshold_weekly"]].values,
        ),
        row=1,
        col=2,
    )

    # Update axis titles.
    fig.update_xaxes(title_text="Rolling Window (Days)", row=1, col=1)
    fig.update_xaxes(title_text="Rolling Window (Weeks)", row=1, col=2)
    fig.update_yaxes(title_text="Ticker", automargin=True)

    fig.update_layout(title=title, margin=dict(l=100, r=50, t=100, b=50))

    fig.show()
    return fig


# Example usage:
# ticker_params = {
#     "AAPL": {"window_daily": 25, "z_threshold_daily": 1.7, "weight_daily": 0.5,
#              "window_weekly": 30, "z_threshold_weekly": 1.9, "weight_weekly": 0.55},
#     "MSFT": {"window_daily": 20, "z_threshold_daily": 1.6, "weight_daily": 0.6,
#              "window_weekly": 28, "z_threshold_weekly": 1.8, "weight_weekly": 0.6},
#     "JPM":  {"window_daily": 30, "z_threshold_daily": 1.8, "weight_daily": 0.55,
#              "window_weekly": 35, "z_threshold_weekly": 2.0, "weight_weekly": 0.5},
#     "BAC":  {"window_daily": 32, "z_threshold_daily": 1.9, "weight_daily": 0.5,
#              "window_weekly": 34, "z_threshold_weekly": 2.1, "weight_weekly": 0.5}
# }
# fig = plot_reversion_bubble(ticker_params, title="Mean Reversion Parameters")
