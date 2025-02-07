import numpy as np
import pandas as pd
import optuna
import hdbscan
from correlation.correlation_utils import compute_correlation_matrix
from utils import logger


def evaluate_clusters(
    returns_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    max_cluster_fraction: float = 0.5,
) -> float:
    """
    Evaluate the clustering quality by calculating the average intra-cluster correlation.
    Penalize results if any cluster (excluding noise) is too large.

    Args:
        returns_df (pd.DataFrame): Asset returns (dates as index, assets as columns).
        cluster_labels (np.ndarray): Cluster labels from the clustering algorithm.
        max_cluster_fraction (float): Maximum allowable fraction of assets in a cluster.

    Returns:
        float: Average intra-cluster correlation, or -1.0 if a giant cluster is found.
    """
    # Exclude noise points (label == -1)
    valid_labels = cluster_labels[cluster_labels != -1]
    if len(valid_labels) == 0:
        return -1.0  # No valid clusters

    total_assets = returns_df.shape[1]
    unique_labels, counts = np.unique(valid_labels, return_counts=True)

    # Penalize if any cluster is too large
    if np.any(counts / total_assets > max_cluster_fraction):
        return -1.0

    correlations = []
    # Calculate intra-cluster correlations for clusters with more than one asset
    for label in unique_labels:
        assets = returns_df.columns[cluster_labels == label]
        if len(assets) > 1:
            corr_matrix = returns_df[assets].corr().values
            # Extract the upper-triangle values (excluding the diagonal)
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            correlations.extend(corr_matrix[mask])

    return np.mean(correlations) if correlations else -1.0


def objective_hdbscan_decorrelation(
    trial: optuna.Trial, returns_df: pd.DataFrame
) -> float:
    """
    Args:
        trial (optuna.Trial): Current Optuna trial.
        returns_df (pd.DataFrame): DataFrame with dates as index and asset returns as columns.

    Returns:
        float: The average intra-cluster correlation as the quality metric (to maximize).
    """
    # Suggest a minimum cluster size between 2 and 10.
    min_cluster_size = trial.suggest_int("min_cluster_size", 2, 10)

    # Instead of allowing min_samples to be low (which may allow giant, loosely connected clusters),
    # require that min_samples is a high fraction of min_cluster_size.
    min_samples_fraction = trial.suggest_float(
        "min_samples_fraction", 0.8, 1.0, step=0.05
    )
    min_samples = int(np.ceil(min_cluster_size * min_samples_fraction))

    # Suggest a cluster_selection_epsilon parameter (0 means default behavior)
    cluster_selection_epsilon = trial.suggest_float(
        "cluster_selection_epsilon", 0.0, 0.3, step=0.05
    )

    logger.info(
        f"Trial {trial.number}: Testing parameters: "
        f"min_cluster_size = {min_cluster_size}, min_samples = {min_samples}, "
        f"cluster_selection_epsilon = {cluster_selection_epsilon}"
    )

    # Compute the correlation matrix and derive a distance matrix (distance = 1 - correlation)
    corr_matrix = compute_correlation_matrix(returns_df)
    distance_matrix = 1 - corr_matrix

    # Run HDBSCAN with the suggested parameters
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Evaluate clusters: this function penalizes giant clusters
    quality = evaluate_clusters(returns_df, cluster_labels, max_cluster_fraction=0.5)
    if quality < 0:
        logger.info(
            f"Trial {trial.number}: Invalid clustering (e.g., giant cluster). Quality = {quality}"
        )
        return -1.0  # Penalize invalid clustering outcomes

    logger.info(
        f"Trial {trial.number}: Average intra-cluster correlation = {quality:.4f}"
    )
    return quality


def run_hdbscan_decorrelation_study(
    returns_df: pd.DataFrame, n_trials: int = 50
) -> dict:
    """
    Run an Optuna study to optimize HDBSCAN parameters for clustering asset returns.
    The goal is to produce tight clusters (with high intra‑cluster correlations) without one giant cluster.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and asset returns as columns.
        n_trials (int): Number of Optuna trials to run.

    Returns:
        dict: Best parameters found by the study.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_hdbscan_decorrelation(trial, returns_df),
        n_trials=n_trials,
    )

    best_params = study.best_trial.params
    logger.info(f"Best params: {best_params} with quality {study.best_value:.4f}")
    return best_params
