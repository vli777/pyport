import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_reversion_params(ticker_params: dict, title: str = "Reversion Parameters"):
    """
    Plot rolling windows and z-thresholds for each ticker in two separate subplots.

    Top: Rolling windows for daily and weekly signals.
    Bottom: Z-thresholds for daily and weekly signals.

    Args:
        ticker_params (dict): Dictionary keyed by ticker with parameters, e.g.
            {
                'AAPL': {
                    'window_daily': 25,
                    'window_weekly': 25,
                    'z_threshold_daily': 1.7,
                    'z_threshold_weekly': 1.7,
                    'group': '1'  (optional)
                },
                ...
            }
        title (str): Overall title for the figure.

    Returns:
        plotly.graph_objects.Figure: The resulting figure.
    """
    # Convert the dictionary into a DataFrame.
    df = pd.DataFrame.from_dict(ticker_params, orient="index")
    df.index.name = "ticker"
    df.reset_index(inplace=True)

    # Optional: if group information is not in the dictionary, you can assign a default value.
    if "group" not in df.columns:
        df["group"] = "All"

    # Sort the DataFrame by group then ticker (or by window if desired)
    df = df.sort_values(by=["group", "ticker"])

    # Create subplots: 2 rows, 1 column.
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Rolling Windows", "Z-Thresholds"),
    )

    # Top subplot: Rolling windows.
    # Create two bar traces: one for daily and one for weekly windows.
    fig.add_trace(
        go.Bar(
            x=df["ticker"],
            y=df["window_daily"],
            name="Daily Window",
            marker_color="steelblue",
            hovertemplate="Ticker: %{x}<br>Daily Window: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["ticker"],
            y=df["window_weekly"],
            name="Weekly Window",
            marker_color="lightseagreen",
            hovertemplate="Ticker: %{x}<br>Weekly Window: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Bottom subplot: Z-thresholds.
    # Create two bar traces: one for daily and one for weekly thresholds.
    fig.add_trace(
        go.Bar(
            x=df["ticker"],
            y=df["z_threshold_daily"],
            name="Daily Z-Threshold",
            marker_color="indianred",
            hovertemplate="Ticker: %{x}<br>Daily Threshold: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["ticker"],
            y=df["z_threshold_weekly"],
            name="Weekly Z-Threshold",
            marker_color="darkorange",
            hovertemplate="Ticker: %{x}<br>Weekly Threshold: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update layout to show x-axis labels only on the bottom subplot.
    fig.update_xaxes(title_text="Ticker", row=2, col=1)
    fig.update_yaxes(title_text="Window Size", row=1, col=1)
    fig.update_yaxes(title_text="Z-Threshold", row=2, col=1)

    # Improve overall layout.
    fig.update_layout(
        barmode="group",  # bars for each ticker are shown side by side.
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50),
    )

    fig.show()
    return fig


def plot_reversion_clusters(
    returns_df: pd.DataFrame,
    group_reversion_signals: dict,
    optimal_period_weights: dict,
    title: str = "Reversion Clusters Visualization",
):
    """
    Plot a t-SNE visualization of the groups obtained from clustering,
    including the tickers that belong to each group and annotating with
    the group's optimized parameters.

    Args:
        returns_df (pd.DataFrame): Returns data with dates as index and tickers as columns.
        group_reversion_signals (dict): Dictionary keyed by group labels with structure:
            {
              group_label: {
                  "tickers": [list of tickers],
                  "daily": {ticker: {date: signal}, ...},
                  "weekly": {ticker: {date: signal}, ...}
              },
              ...
            }
        optimal_period_weights (dict): Mapping from group label to optimized parameters.
            For example: { group_label: {"weight_daily": 0.5, "weight_weekly": 0.5}, ... }
        title (str): Title of the plot.

    Returns:
        plotly.graph_objects.Figure: The t-SNE scatter plot figure.
    """
    # Build a DataFrame with ticker and group label.
    records = []
    for group_label, group_data in group_reversion_signals.items():
        for ticker in group_data.get("tickers", []):
            records.append(
                {
                    "ticker": ticker,
                    "group": str(group_label),
                    # Optionally include optimized parameters in the record for hover text.
                    "weight_daily": optimal_period_weights.get(group_label, {}).get(
                        "weight_daily", np.nan
                    ),
                    "weight_weekly": optimal_period_weights.get(group_label, {}).get(
                        "weight_weekly", np.nan
                    ),
                }
            )

    if not records:
        print("No groups/tickers to plot.")
        return None

    df_plot = pd.DataFrame(records)

    # Subset returns_df to only tickers that are in df_plot
    tickers_to_plot = df_plot["ticker"].unique().tolist()
    returns_sub = returns_df[tickers_to_plot].dropna(axis=1, how="all")

    # Use t-SNE on the transposed returns data (each ticker is a sample)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(returns_sub.T.fillna(0))

    tsne_df = pd.DataFrame(embeddings, columns=["x", "y"])
    tsne_df["ticker"] = returns_sub.columns
    # Merge with our group information
    tsne_df = tsne_df.merge(df_plot, on="ticker", how="left")

    # Create a scatter plot, coloring by group
    fig = px.scatter(
        tsne_df,
        x="x",
        y="y",
        color="group",
        hover_data={
            "ticker": True,
            "weight_daily": True,
            "weight_weekly": True,
            "group": False,
        },
        title=title,
    )

    # Optionally, annotate each cluster with its tickers (as a comma-separated list)
    # You could also add other annotations here as needed.
    cluster_annotations = {}
    for group_label, group_df in tsne_df.groupby("group"):
        tickers_in_group = ", ".join(group_df["ticker"].tolist())
        # Compute the centroid of the group's points for annotation placement.
        centroid_x = group_df["x"].mean()
        centroid_y = group_df["y"].mean()
        annotation_text = f"Group {group_label}<br>Tickers: {tickers_in_group}"
        # Optionally, append optimized weights to the annotation.
        if group_label in optimal_period_weights:
            w_daily = optimal_period_weights[group_label].get("weight_daily", np.nan)
            w_weekly = optimal_period_weights[group_label].get("weight_weekly", np.nan)
            annotation_text += f"<br>Daily: {w_daily:.2f}, Weekly: {w_weekly:.2f}"
        cluster_annotations[group_label] = (centroid_x, centroid_y, annotation_text)
        fig.add_annotation(
            x=centroid_x,
            y=centroid_y,
            text=annotation_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
        )

    fig.update_layout(title=title)
    fig.show()
    return fig
