import math
import numpy as np
import pandas as pd
import optuna
import hdbscan
from sklearn.metrics import silhouette_score

from correlation.correlation_utils import compute_distance_matrix
from utils.logger import logger


def objective_hdbscan_decorrelation(
    trial: optuna.Trial, returns_df: pd.DataFrame, scale_distances: bool = False
) -> float:
    """
    Objective function for optimizing HDBSCAN parameters.
    Maximizes the silhouette score while penalizing solutions with overly large clusters
    or too few clusters. In addition to epsilon, alpha, and cluster_selection_epsilon_max,
    this function also tunes min_cluster_size and min_samples.
    """
    # Tune basic HDBSCAN parameters.
    epsilon = trial.suggest_float("epsilon", 0.15, 0.3, step=0.01)
    alpha = trial.suggest_float("alpha", 0.1, 2.0, step=0.01)
    cluster_selection_epsilon_max = trial.suggest_float(
        "cluster_selection_epsilon_max", 0.3, 1.0, step=0.01
    )

    # Define a range for min_cluster_size based on number of assets.
    n_assets = returns_df.shape[1]
    # max_cluster_size = max(2, min(n_assets // 3, 20))
    # min_cluster_size = trial.suggest_int("min_cluster_size", 2, max_cluster_size)
    # Allow min_samples to vary from 1 up to min_cluster_size.
    # min_samples = trial.suggest_int("min_samples", 1, min_cluster_size)

    # Compute (and optionally scale) the distance matrix.
    distance_matrix = compute_distance_matrix(
        returns_df, scale_distances=scale_distances
    )

    # For reproducibility.
    np.random.seed(42)
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        alpha=alpha,
        min_cluster_size=2,
        # min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf",
        cluster_selection_epsilon_max=cluster_selection_epsilon_max,
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Only consider non-noise points.
    valid_idx = cluster_labels != -1
    valid_labels = cluster_labels[valid_idx]
    unique_labels = np.unique(valid_labels)
    if len(unique_labels) < 2:
        logger.debug(
            "Trial rejected: less than 2 clusters (insufficient valid clusters)."
        )
        return -1.0

    # Convert the distance matrix to a NumPy array (if not already) and filter for valid indices.
    # This avoids pandas indexing issues when using boolean arrays.
    distance_matrix = np.asarray(distance_matrix)
    valid_distance_matrix = distance_matrix[valid_idx][:, valid_idx].copy()
    np.fill_diagonal(valid_distance_matrix, 0)
    try:
        sil_score = silhouette_score(
            valid_distance_matrix, valid_labels, metric="precomputed"
        )
    except Exception as e:
        logger.debug(f"Silhouette score computation failed: {e}")
        return -1.0

    # Penalize solutions with one giant cluster or too few clusters.
    sqrt_n = math.ceil(math.sqrt(n_assets))
    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
    max_cluster_fraction = max(cluster_sizes) / n_assets
    num_clusters = len(unique_labels)

    penalty = 0.0
    threshold_fraction = (
        0.05  # Allow clusters up to 5% of total assets without penalty.
    )
    penalty_weight = 0.5
    if max_cluster_fraction > threshold_fraction:
        penalty += penalty_weight * (max_cluster_fraction - threshold_fraction)
    if num_clusters < sqrt_n:
        penalty += penalty_weight * (sqrt_n - num_clusters)

    overall_quality = sil_score - penalty

    logger.debug(
        f"Trial params: epsilon={epsilon}, alpha={alpha}, eps_max={cluster_selection_epsilon_max}, "
        # f"min_cluster_size={min_cluster_size}, min_samples={min_samples} | "
        f"Silhouette: {sil_score:.4f}, Penalty: {penalty:.4f}, Overall: {overall_quality:.4f}"
    )
    return overall_quality


def run_hdbscan_decorrelation_study(
    returns_df: pd.DataFrame, n_trials: int = 50, scale_distances: bool = False
) -> dict:
    """
    Run an Optuna study to optimize HDBSCAN parameters.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_hdbscan_decorrelation(
            trial, returns_df, scale_distances=scale_distances
        ),
        n_trials=n_trials,
    )
    best_params = study.best_trial.params
    logger.info(f"Best parameters: {best_params} with quality {study.best_value:.4f}")
    return best_params
