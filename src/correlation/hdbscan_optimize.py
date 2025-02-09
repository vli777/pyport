import math
import numpy as np
import pandas as pd
import optuna
import hdbscan
from sklearn.metrics import silhouette_score

from correlation.correlation_utils import compute_correlation_matrix
from utils.logger import logger


def objective_hdbscan_decorrelation(trial: optuna.Trial, returns_df) -> float:
    """
    Optimize HDBSCAN parameters for clustering stock returns.
    The goal is to maximize the silhouette score (which balances low intra-cluster
    distances and high inter-cluster separation) while penalizing clusters that are
    too large (which may indicate that obvious subgroups like semiconductors are merged).
    """
    # Tune parameters over a broad range.
    epsilon = trial.suggest_float("epsilon", 0.0, 0.3, step=0.01)
    alpha = trial.suggest_float("alpha", 0.1, 1.0, step=0.01)
    cluster_selection_epsilon_max = trial.suggest_float(
        "cluster_selection_epsilon_max", 0.3, 0.5, step=0.01
    )

    # Compute the correlation matrix and convert it to a normalized distance matrix [0,1]
    corr_matrix = compute_correlation_matrix(returns_df)  # Your user-defined function
    # Convert to a NumPy array (in case it is a DataFrame) to ease indexing:
    corr_np = (
        corr_matrix.to_numpy()
        if hasattr(corr_matrix, "to_numpy")
        else np.array(corr_matrix)
    )
    distance_matrix = (
        1 - corr_np
    ) / 2  # 0 when perfectly correlated, 1 when perfectly anti-correlated

    # Run HDBSCAN clustering on the precomputed distance matrix.
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        alpha=alpha,
        min_cluster_size=2,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf",
        cluster_selection_epsilon_max=cluster_selection_epsilon_max,
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Filter out noise (label = -1)
    valid = cluster_labels != -1
    valid_labels = cluster_labels[valid]
    unique_labels = np.unique(valid_labels)

    # Require at least 2 clusters to compute a silhouette score.
    if len(unique_labels) < 2:
        return -1.0

    # Compute silhouette score on the non-noise subset.
    # Use a copy of the valid submatrix and zero out the diagonal.
    valid_distance = distance_matrix[valid][:, valid].copy()
    np.fill_diagonal(valid_distance, 0)
    try:
        sil_score = silhouette_score(valid_distance, valid_labels, metric="precomputed")
    except Exception:
        return -1.0

    # --- Penalize large clusters ---
    # Compute cluster sizes and determine the fraction of stocks in the largest cluster.
    n_assets = returns_df.shape[1]
    sqrt_n = math.ceil(math.sqrt(n_assets))

    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
    max_cluster_fraction = max(cluster_sizes) / n_assets
    num_clusters = len(unique_labels)
    # For example, if any cluster contains more than 10% of all stocks,
    # apply a penalty. Adjust threshold and weight as needed.
    penalty = 0.0
    threshold = 0.05  # Allow clusters up to 2% of total stocks without penalty.
    penalty_weight = 0.5  # Weight of the penalty term.
    if max_cluster_fraction > threshold:
        penalty += penalty_weight * (max_cluster_fraction - threshold)
    if num_clusters < sqrt_n:
        penalty += penalty_weight * (sqrt_n - num_clusters)

    overall_quality = sil_score - penalty

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
