import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px


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
