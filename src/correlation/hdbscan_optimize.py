import numpy as np
import pandas as pd
import optuna
import hdbscan
import math

from correlation.correlation_utils import compute_correlation_matrix
from utils.logger import logger


def evaluate_clusters(
    returns_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    max_cluster_fraction: float = 0.5,
) -> float:
    """
    Evaluate clustering quality by calculating average intra-cluster correlation.
    Penalize if any cluster (excluding noise) is too large or if there are too few clusters.
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

    # Penalize if there are too few clusters
    min_cluster_count = max(3, math.ceil(math.sqrt(total_assets) / 2))
    if len(unique_labels) < min_cluster_count:
        return -1.0

    correlations = []
    for label in unique_labels:
        assets = returns_df.columns[cluster_labels == label]
        if len(assets) > 1:
            corr_matrix = returns_df[assets].corr().values
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            correlations.extend(corr_matrix[mask])

    return np.mean(correlations) if correlations else -1.0


def objective_hdbscan_decorrelation(
    trial: optuna.Trial, returns_df: pd.DataFrame
) -> float:
    """
    Optimize HDBSCAN parameters for clustering asset returns.
    This function tunes:
      - min_cluster_size over a range surrounding the √(n_assets) heuristic,
      - min_samples (as a fraction of min_cluster_size), and
      - cluster_selection_epsilon.
    """
    n_assets = returns_df.shape[1]
    # Compute baseline min_cluster_size using √(n_assets)
    sqrt_n = math.ceil(math.sqrt(n_assets))
    # Define a range around the baseline—for example, ±3 (ensuring a lower bound of 3)
    min_cluster_size_lower = max(3, sqrt_n - 3)
    min_cluster_size_upper = sqrt_n + 3
    min_cluster_size = trial.suggest_int(
        "min_cluster_size", min_cluster_size_lower, min_cluster_size_upper
    )

    # Tune min_samples as a fraction of min_cluster_size
    min_samples_fraction = trial.suggest_float(
        "min_samples_fraction", 0.8, 1.0, step=0.05
    )
    min_samples = int(np.ceil(min_cluster_size * min_samples_fraction))

    # Tune cluster_selection_epsilon over a wider range
    cluster_selection_epsilon = trial.suggest_float(
        "cluster_selection_epsilon", 0.0, 1.0, step=0.1
    )

    logger.info(
        f"Trial {trial.number}: Testing parameters: "
        f"min_cluster_size = {min_cluster_size} (range: {min_cluster_size_lower}-{min_cluster_size_upper}), "
        f"min_samples = {min_samples}, "
        f"cluster_selection_epsilon = {cluster_selection_epsilon}"
    )

    # Compute correlation and distance matrices
    corr_matrix = compute_correlation_matrix(returns_df)
    distance_matrix = 1 - corr_matrix

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    quality = evaluate_clusters(
        returns_df, cluster_labels, max_cluster_fraction=0.5, min_cluster_count=5
    )
    if quality < 0:
        logger.info(
            f"Trial {trial.number}: Invalid clustering (giant cluster or too few clusters). Quality = {quality}"
        )
        return -1.0
    logger.info(
        f"Trial {trial.number}: Average intra-cluster correlation = {quality:.4f}"
    )
    return quality


def run_hdbscan_decorrelation_study(
    returns_df: pd.DataFrame, n_trials: int = 50
) -> dict:
    """
    Run an Optuna study to optimize HDBSCAN parameters.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_hdbscan_decorrelation(trial, returns_df),
        n_trials=n_trials,
    )

    best_params = study.best_trial.params
    logger.info(f"Best params: {best_params} with quality {study.best_value:.4f}")
    return best_params
