from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
import plotly.express as px

from correlation.correlation_utils import compute_ticker_hash
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle
from utils.performance_metrics import kappa_ratio, sharpe_ratio
from utils.logger import logger


def filter_correlated_groups_dbscan(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    eps: float = 0.2,
    min_samples: int = 2,
    top_n_per_cluster: int = 1,
    plot: bool = False,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
) -> list:
    """
    Uses DBSCAN to cluster assets based on the distance (1 - correlation) matrix.
    Then, for each cluster, selects the top performing stock(s) based on a composite
    performance metric computed internally.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        risk_free_rate (float): Risk-free rate for performance metric calculation.
        eps (float): The eps parameter for DBSCAN (distance threshold).
        min_samples (int): Minimum samples for a core point in DBSCAN.
        top_n_per_cluster (int): How many top assets to select from each cluster.
        plot (bool): If True, display a t-SNE visualization of clusters.
        cache_dir (str): Directory path to cache optimized DBSCAN parameters.
        reoptimize (bool): If True, force re-optimization (or re-calculation) of eps.

    Returns:
        list: A list of selected ticker symbols after decorrelation.
    """
    # Optionally load cached eps (if optimizing and caching)
    # Ensure the cache directory exists
    cache_path = Path(cache_dir)
    cache_path.mkdir(
        parents=True, exist_ok=True
    )  # Create the directory if it doesn't exist

    # Compute unique cache filename based on tickers
    tickers = returns_df.columns.tolist()
    ticker_hash = compute_ticker_hash(tickers)
    cache_filename = cache_path / f"dbscan_epsilon_{ticker_hash}.pkl"  # Path object

    cached_params = load_parameters_from_pickle(cache_filename)
    if not reoptimize and "eps" in cached_params:
        eps = cached_params["eps"]
    else:
        cached_params["eps"] = eps
        save_parameters_to_pickle(cached_params, cache_filename)

    # Compute correlation matrix (robust if needed)
    if returns_df.shape[1] > 50:
        corr_matrix = compute_lw_correlation(returns_df)
    else:
        corr_matrix = returns_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 1.0)

    # Convert correlation to distance: distance = 1 - correlation.
    distance_matrix = 1 - corr_matrix

    # Cluster assets with DBSCAN using the precomputed distance matrix.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(distance_matrix)

    # Group tickers by cluster label.
    clusters = {}
    for ticker, label in zip(returns_df.columns, cluster_labels):
        clusters.setdefault(label, []).append(ticker)

    logger.info(f"Total clusters found: {len(clusters)}")

    # Compute performance metrics internally.
    perf_series = compute_performance_metrics(returns_df, risk_free_rate)

    # Select best-performing ticker(s) in each cluster.
    selected_tickers = []
    for label, tickers in clusters.items():
        # If label == -1 (noise), simply include them.
        if label == -1:
            selected_tickers.extend(tickers)
        else:
            # From each cluster, sort by composite score.
            group_perf = perf_series[tickers].sort_values(ascending=False)
            top_candidates = group_perf.index.tolist()[:top_n_per_cluster]
            selected_tickers.extend(top_candidates)
            logger.info(
                f"Cluster {label}: {len(tickers)} assets: Keeping {top_candidates}"
            )

    removed_tickers = set(returns_df.columns) - set(selected_tickers)
    if removed_tickers:
        logger.info(
            f"Removed {len(removed_tickers)} assets due to high correlation: {sorted(removed_tickers)}"
        )
    else:
        logger.info("No assets were removed.")
    logger.info(f"{len(selected_tickers)} assets remain")

    # Optional t-SNE visualization.
    if plot:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(returns_df.T)
        tsne_df = pd.DataFrame(embeddings, columns=["x", "y"])
        tsne_df["ticker"] = returns_df.columns
        tsne_df["cluster"] = cluster_labels.astype(str)
        fig = px.scatter(
            tsne_df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["ticker"],
            title="t-SNE Visualization of Stock Clusters (DBSCAN)",
        )
        fig.show()

        # --- k-distance plot for eps selection ---

        # Use the precomputed distance matrix.
        # NearestNeighbors with metric='precomputed' requires the full distance matrix.
        nbrs = NearestNeighbors(n_neighbors=min_samples, metric="precomputed")
        nbrs.fit(distance_matrix)
        distances, _ = nbrs.kneighbors(distance_matrix)

        # We want the distance to the min_samples-th nearest neighbor.
        kth_distances = np.sort(distances[:, -1])

        # Create a line plot using Plotly.
        k_values = list(range(1, len(kth_distances) + 1))
        fig_k = px.line(
            x=k_values,
            y=kth_distances,
            labels={
                "x": "Points sorted by distance",
                "y": f"Distance to {min_samples}-th nearest neighbor",
            },
            title="k-Distance Plot for eps Selection",
        )

        # Add a horizontal line at the selected eps value.
        fig_k.add_hline(
            y=eps,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Selected eps: {eps}",
            annotation_position="top right",
        )

        fig_k.show()

    return selected_tickers


def compute_lw_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a robust correlation matrix using the Ledoit-Wolf covariance estimate.

    Args:
        df (pd.DataFrame): DataFrame of returns with assets as columns.

    Returns:
        pd.DataFrame: Correlation matrix with assets as both index and columns.
    """
    lw = LedoitWolf()
    covariance = lw.fit(df).covariance_
    std_dev = np.sqrt(np.diag(covariance))
    corr = covariance / np.outer(std_dev, std_dev)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=df.columns, columns=df.columns)


def compute_performance_metrics(
    returns_df: pd.DataFrame, risk_free_rate: float
) -> pd.Series:
    metrics = {}
    for ticker in returns_df.columns:
        ticker_returns = returns_df[ticker].dropna()
        if ticker_returns.empty:
            continue
        cumulative_return = (ticker_returns + 1).prod() - 1
        sr = sharpe_ratio(ticker_returns, risk_free_rate)
        kp = kappa_ratio(ticker_returns, order=3)
        composite = 0.4 * cumulative_return + 0.3 * sr + 0.3 * kp
        metrics[ticker] = composite
    return pd.Series(metrics)
