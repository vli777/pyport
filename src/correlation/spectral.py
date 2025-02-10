import math
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def compute_affinity_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the correlation matrix for the asset returns and convert it
    into an affinity matrix in [0, 1]. A simple way is to map correlation values
    from [-1, 1] to [0, 1] via: affinity = (corr + 1) / 2.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.

    Returns:
        np.ndarray: The affinity matrix.
    """
    corr_matrix = returns_df.corr().values
    # Map correlation from [-1, 1] to [0, 1]
    affinity = (corr_matrix + 1) / 2.0
    return affinity


def compute_performance_metrics(
    returns_df: pd.DataFrame, risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Placeholder for your performance metric computation.
    Replace this with your actual function.

    For example, you might compute a Sharpe ratio or composite score for each asset.
    """
    # Here we simply compute mean returns as a stand-in.
    perf = returns_df.mean()
    return perf


def visualize_clusters_tsne(returns_df: pd.DataFrame, cluster_labels: np.ndarray):
    """
    Placeholder for your TSNE visualization.
    Replace this with your actual plotting function.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(returns_df.T.values)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap="tab20", s=50
    )
    plt.title("TSNE of Asset Returns Clusters")
    plt.colorbar(scatter)
    plt.show()


def filter_correlated_groups_spectral(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_clusters: Optional[int] = None,
    top_n_per_cluster: int = 1,
    plot: bool = False,
) -> List[str]:
    """
    Uses spectral clustering on the asset returns to create fine‑grained, tight clusters.
    The affinity is computed from the correlation matrix (scaled to [0, 1]).

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        risk_free_rate (float): Risk-free rate for performance metric calculation.
        n_clusters (int, optional): The number of clusters to form. If None, it is set to 10% of assets.
        top_n_per_cluster (int): How many top assets to select from each cluster.
        plot (bool): If True, display a TSNE visualization of the clusters.

    Returns:
        List[str]: A list of selected asset tickers.
    """
    n_assets = returns_df.shape[1]
    # If n_clusters is not specified, set it to roughly 10% of the assets.
    if n_clusters is None:
        n_clusters = max(2, math.ceil(n_assets * 0.1))
        logger.info(
            f"n_clusters not provided. Setting n_clusters = {n_clusters} for {n_assets} assets."
        )

    affinity = compute_affinity_matrix(returns_df)

    # Run spectral clustering using the precomputed affinity matrix.
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    cluster_labels = spectral.fit_predict(affinity)

    # Group tickers by cluster label.
    clusters = {}
    tickers = returns_df.columns.tolist()
    for ticker, label in zip(tickers, cluster_labels):
        clusters.setdefault(label, []).append(ticker)
    logger.info(f"Spectral clustering produced {len(clusters)} clusters.")

    # Compute performance metrics (replace with your own function)
    perf_series = compute_performance_metrics(returns_df, risk_free_rate)

    # For each cluster, select the top performer(s)
    selected_tickers: List[str] = []
    for label, ticker_group in clusters.items():
        # You can handle noise (if any) separately; here we assume no special noise label.
        group_perf = perf_series[ticker_group].sort_values(ascending=False)
        top_candidates = group_perf.index.tolist()[:top_n_per_cluster]
        selected_tickers.extend(top_candidates)
        logger.info(
            f"Cluster {label}: {len(ticker_group)} assets; selected {top_candidates}"
        )

    if plot:
        visualize_clusters_tsne(returns_df, cluster_labels)

    removed_tickers = set(tickers) - set(selected_tickers)
    logger.info(
        f"Removed {len(removed_tickers)} assets; {len(selected_tickers)} assets remain."
    )

    return selected_tickers
