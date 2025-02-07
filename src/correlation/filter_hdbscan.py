from pathlib import Path
from datetime import date
from typing import List, Union
import numpy as np
import pandas as pd
import hdbscan
import plotly.express as px
from sklearn.manifold import TSNE
from correlation.correlation_utils import compute_correlation_matrix
from correlation.hdbscan_optimize import run_hdbscan_decorrelation_study
from correlation.tsne_dbscan import compute_performance_metrics
from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def filter_correlated_groups_hdbscan(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    min_cluster_size: Union[int, None] = None,
    min_samples: int = 2,
    top_n_per_cluster: int = 1,
    plot: bool = False,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
) -> list[str]:
    """
    Uses HDBSCAN to cluster assets based on the distance (1 - correlation) matrix.
    Then, for each cluster, selects the top performing asset(s) based on a composite
    performance metric computed internally.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        risk_free_rate (float): Risk-free rate for performance metric calculation.
        min_cluster_size (int, optional): The minimum cluster size for HDBSCAN.
        min_samples (int): Minimum samples for a core point in HDBSCAN.
        top_n_per_cluster (int): How many top assets to select from each cluster.
        plot (bool): If True, display a visualization of clusters.
        cache_dir (str): Directory path to cache optimized HDBSCAN parameters.
        reoptimize (bool): If True, force re-optimization of HDBSCAN parameters.

    Returns:
        list(str): A list of selected ticker symbols after decorrelation.
    """
    # Ensure the cache directory exists
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create a cache filename (this can be based on date range, etc.)
    # start_date = returns_df.index.min().strftime("%Y%m%d")
    # end_date = returns_df.index.max().strftime("%Y%m%d")
    cache_filename = cache_path / "hdbscan_params.pkl"
    cached_params = load_parameters_from_pickle(cache_filename) or {}

    # Use cached parameters if available, otherwise reoptimize
    if all(
        param in cached_params
        for param in [
            "min_cluster_size",
            "min_samples",
            "cluster_selection_epsilon",
        ]
    ):
        min_cluster_size = cached_params["min_cluster_size"]
        min_samples = cached_params["min_samples"]
        cluster_selection_epsilon = cached_params["cluster_selection_epsilon"]
    else:
        reoptimize = True

    if reoptimize:
        best_params = run_hdbscan_decorrelation_study(
            returns_df=returns_df, n_trials=50
        )
        min_cluster_size = best_params["min_cluster_size"]
        min_samples_fraction = best_params["min_samples_fraction"]
        min_samples = int(np.ceil(min_cluster_size * min_samples_fraction))
        cluster_selection_epsilon = best_params["cluster_selection_epsilon"]

        cached_params["min_cluster_size"] = min_cluster_size
        cached_params["min_samples"] = min_samples
        cached_params["cluster_selection_epsilon"] = cluster_selection_epsilon

        save_parameters_to_pickle(cached_params, cache_filename)

    # Compute the correlation matrix
    corr_matrix = compute_correlation_matrix(returns_df)
    # Convert correlation to distance: distance = 1 - correlation.
    distance_matrix = 1 - corr_matrix

    # Cluster assets with HDBSCAN using the precomputed distance matrix.
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Group tickers by cluster label.
    clusters = {}
    for ticker, label in zip(returns_df.columns, cluster_labels):
        clusters.setdefault(label, []).append(ticker)

    logger.info(f"Total clusters found: {len(clusters)}")

    # Compute performance metrics (this function should compute a composite score or similar)
    perf_series = compute_performance_metrics(returns_df, risk_free_rate)

    # Select best-performing ticker(s) in each cluster.
    selected_tickers: List[str] = []
    for label, tickers in clusters.items():
        # If label == -1 (noise), simply include them.
        if label == -1:
            selected_tickers.extend(tickers)
        else:
            group_perf = perf_series[tickers].sort_values(ascending=False)
            top_candidates = group_perf.index.tolist()[:top_n_per_cluster]
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

    if plot:
        visualize_clusters_tsne(returns_df=returns_df, cluster_labels=cluster_labels)
    return selected_tickers


def visualize_clusters_tsne(
    returns_df: pd.DataFrame, cluster_labels, perplexity: int = 30, max_iter: int = 1000
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
    tsne = TSNE(perplexity=perplexity, max_iter=max_iter, random_state=42)
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
