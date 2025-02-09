import numpy as np
import pandas as pd
import optuna
import hdbscan
import math
from sklearn.metrics import silhouette_score

from correlation.correlation_utils import compute_correlation_matrix
from utils.logger import logger


def evaluate_clusters(
    returns_df: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> float:
    """
    Evaluate clustering quality by computing the average intra-cluster correlation.
    """
    # Exclude noise points
    valid_labels = cluster_labels[cluster_labels != -1]
    if len(valid_labels) == 0:
        return -1.0  # No valid clusters

    total_assets = returns_df.shape[1]
    unique_labels, counts = np.unique(valid_labels, return_counts=True)

    # Penalize if there are too few clusters overall
    min_cluster_count = math.ceil(math.sqrt(total_assets))
    if len(unique_labels) < min_cluster_count / 2:
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
      - min_samples, and
      - epsilon.
    """
    n_assets = returns_df.shape[1]
    sqrt_n = math.ceil(math.sqrt(n_assets))

    min_samples = trial.suggest_int("min_samples", 2, sqrt_n, step=1)
    epsilon = trial.suggest_float("epsilon", 0.0, 1.0, step=0.01)

    # Compute correlation and normalized distance matrix (0 to 1)
    corr_matrix = compute_correlation_matrix(returns_df)
    distance_matrix = (1 - corr_matrix) / 2

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=2,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf",
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    quality_corr = evaluate_clusters(returns_df, cluster_labels)
    if quality_corr < 0:
        logger.info(
            f"Trial {trial.number}: Invalid clustering. Quality_corr={quality_corr}"
        )
        return -1.0

    # Compute silhouette score on non-noise points
    valid_indices = cluster_labels != -1
    unique_labels = np.unique(cluster_labels[valid_indices])
    if len(unique_labels) > 1:
        sil_score = silhouette_score(
            distance_matrix[valid_indices][:, valid_indices],
            cluster_labels[valid_indices],
            metric="precomputed",
        )
    else:
        sil_score = -1.0

    # Combine the metrics (here we average them; adjust weights as needed)
    overall_quality = (quality_corr + sil_score) / 2.0
    num_clusters = len(unique_labels)
    logger.info(
        f"Trial {trial.number}: Quality_corr={quality_corr:.4f}, Silhouette={sil_score:.4f}, "
        f"Combined Quality={overall_quality:.4f}, Number of clusters={num_clusters}"
    )
    return overall_quality


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
