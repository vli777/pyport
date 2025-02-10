import math
from pathlib import Path
from datetime import date
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from correlation.correlation_utils import compute_correlation_matrix
from correlation.hdbscan_optimize import run_hdbscan_decorrelation_study
from correlation.tsne_dbscan import compute_performance_metrics
from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def filter_correlated_groups_hdbscan(
    returns_df: pd.DataFrame,
    asset_cluster_map: Dict[str, int],
    risk_free_rate: float = 0.0,
    plot: bool = False,
) -> list[str]:
    """
    Uses HDBSCAN to cluster assets based on the distance (1 - correlation) matrix.
    Then, for each cluster, selects the top performing asset(s) based on a composite
    performance metric computed internally.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        asset_cluster_map (dict): Full set of assets mapped to optimized HDBSCAN clusters.
        risk_free_rate (float): Risk-free rate for performance metric calculation.
        plot (bool): If True, display a visualization of clusters.

    Returns:
        list(str): A list of selected ticker symbols after decorrelation.
    """
    # Group tickers by their cluster label
    clusters = {}
    for ticker, label in asset_cluster_map.items():
        clusters.setdefault(label, []).append(ticker)

    # Compute performance metrics for each asset
    perf_series = compute_performance_metrics(returns_df, risk_free_rate)

    # Select the best-performing tickers from each cluster
    selected_tickers: list[str] = []
    for label, tickers in clusters.items():
        # For noise (label == -1), include all tickers
        if label == -1:
            selected_tickers.extend(tickers)
        else:
            group_perf = perf_series[tickers].sort_values(ascending=False)
            if len(tickers) < 10:
                top_n = len(tickers)
            elif len(tickers) < 20:
                top_n = max(1, int(0.50 * len(tickers)))
            else:
                top_n = max(1, int(0.33 * len(tickers)))
            top_candidates = group_perf.index.tolist()[:top_n]
            selected_tickers.extend(top_candidates)
            logger.info(
                f"Cluster {label}: {len(tickers)} assets; keeping {top_candidates}"
            )

    removed_tickers = set(returns_df.columns) - set(selected_tickers)
    if removed_tickers:
        logger.info(
            f"Removed {len(removed_tickers)} assets due to high correlation: {sorted(removed_tickers)}"
        )
    else:
        logger.info("No assets were removed.")
    logger.info(f"{len(selected_tickers)} assets remain")

    # Optionally, visualize the clusters using TSNE
    if plot:
        labels_in_order = np.array(
            [asset_cluster_map[ticker] for ticker in returns_df.columns]
        )
        visualize_clusters_tsne(returns_df=returns_df, cluster_labels=labels_in_order)

    return selected_tickers


def visualize_clusters_tsne(
    returns_df: pd.DataFrame,
    cluster_labels,
    perplexity: float = 30,
    max_iter: int = 1000,
):
    """
    Visualize asset clusters using t-SNE and Plotly.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        cluster_labels (array-like): Cluster labels for each asset (in the same order as returns_df.columns).
        perplexity (int): t-SNE perplexity parameter.
        max_iter (int): Number of iterations for t-SNE.
    """
    # Transpose the DataFrame so that each asset is represented as a feature vector.
    asset_data = returns_df.T

    # Optionally, you can standardize the data here if needed.
    tsne = TSNE(
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=42,
    )
    tsne_results = tsne.fit_transform(asset_data)

    # Create a DataFrame for plotting.
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])
    tsne_df["Ticker"] = asset_data.index
    tsne_df["Cluster"] = cluster_labels

    # Create an interactive scatter plot.
    fig = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color="Cluster",
        hover_data=["Ticker"],
        title="t-SNE Visualization of Asset Clusters",
    )
    fig.show()


def get_cluster_labels(
    returns_df: pd.DataFrame,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
) -> dict[str, int]:
    from pathlib import Path
    import numpy as np
    import hdbscan

    # Load cached parameters
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = cache_path / "hdbscan_params.pkl"
    cached_params = load_parameters_from_pickle(cache_filename) or {}

    if all(
        param in cached_params
        for param in ["epsilon", "alpha", "cluster_selection_epsilon_max"]
    ):
        epsilon = cached_params["epsilon"]
        alpha = cached_params["alpha"]
        cluster_selection_epsilon_max = cached_params["cluster_selection_epsilon_max"]
    else:
        reoptimize = True

    if reoptimize:
        best_params = run_hdbscan_decorrelation_study(
            returns_df=returns_df, n_trials=100
        )
        epsilon = best_params["epsilon"]
        alpha = best_params["alpha"]
        cluster_selection_epsilon_max = best_params["cluster_selection_epsilon_max"]
        cached_params = {
            "epsilon": epsilon,
            "alpha": alpha,
            "cluster_selection_epsilon_max": cluster_selection_epsilon_max,
        }
        save_parameters_to_pickle(cached_params, cache_filename)

    # Compute the correlation matrix and convert it to a normalized distance matrix
    corr_matrix = compute_correlation_matrix(returns_df)
    distance_matrix = (1 - corr_matrix) / 2  # Normalize to 0–1

    # Perform clustering using HDBSCAN with the precomputed distance matrix
    np.random.seed(42)
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        alpha=alpha,
        min_cluster_size=2,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf",
        cluster_selection_epsilon_max=cluster_selection_epsilon_max,
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Map each ticker to its corresponding cluster label
    asset_cluster_map = dict(zip(returns_df.columns, cluster_labels))
    cache_cluster_filename = cache_path / "hdbscan_clusters.pkl"
    save_parameters_to_pickle(asset_cluster_map, cache_cluster_filename)

    # Log the total number of clusters found (ignoring noise labeled as -1)
    labels_in_order = np.array(
        [asset_cluster_map[ticker] for ticker in returns_df.columns]
    )
    num_clusters = len(np.unique(labels_in_order[labels_in_order != -1]))
    logger.info(f"Total clusters found: {num_clusters}")

    return asset_cluster_map
