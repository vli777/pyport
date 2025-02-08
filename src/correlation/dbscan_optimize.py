import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import DBSCAN

from correlation.correlation_utils import compute_correlation_matrix
from utils.logger import logger


def evaluate_dbscan_clusters(
    returns_df: pd.DataFrame, cluster_labels: np.ndarray
) -> float:
    """
    Evaluate DBSCAN clustering quality by computing the average intra-cluster correlation.
    Only clusters with at least two assets are considered. Noise (label == -1) or singletons
    are ignored.

    Returns:
        float: Average intra-cluster correlation (the higher, the better).
               Returns -1.0 if no valid clusters are found.
    """
    corr_matrix = compute_correlation_matrix(returns_df)
    clusters = {}
    for ticker, label in zip(returns_df.columns, cluster_labels):
        clusters.setdefault(label, []).append(ticker)

    intra_corrs = []
    for label, tickers in clusters.items():
        if label == -1 or len(tickers) < 2:
            continue
        sub_corr = corr_matrix.loc[tickers, tickers]
        n = len(tickers)
        # Extract off-diagonal (pairwise) correlation values.
        pairwise_corr = sub_corr.values[np.triu_indices(n, k=1)]
        if pairwise_corr.size > 0:
            intra_corrs.append(pairwise_corr.mean())

    if not intra_corrs:
        return -1.0
    return np.mean(intra_corrs)


def objective_dbscan_decorrelation(
    trial: optuna.Trial, returns_df: pd.DataFrame, min_samples: int = 2
) -> float:
    """
    Optuna objective function to optimize the eps parameter for DBSCAN.
    The goal is to maximize the average intra‑cluster correlation (i.e. each cluster’s
    members are highly correlated), so that when we select only the top performer
    from each cluster, the remaining portfolio consists of decorrelated assets.

    Args:
        trial (optuna.Trial): Current trial.
        returns_df (pd.DataFrame): DataFrame with dates as index and asset returns as columns.
        min_samples (int): Minimum samples for DBSCAN.

    Returns:
        float: The average intra‑cluster correlation as the objective (to maximize).
    """
    # Suggest an eps value in a reasonable range (adjust as needed)
    eps = trial.suggest_float("eps", 0.01, 1.0, log=True)
    logger.info(f"Trial {trial.number}: Testing eps = {eps:.4f}")

    # Compute correlation and derive the distance matrix (distance = 1 - correlation)
    corr_matrix = compute_correlation_matrix(returns_df)
    distance_matrix = 1 - corr_matrix

    # Run DBSCAN clustering with the trial-suggested eps.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(distance_matrix)

    # Evaluate the clustering quality.
    quality = evaluate_dbscan_clusters(returns_df, cluster_labels)
    if quality < 0:
        logger.info(
            f"Trial {trial.number}: No valid clusters found. Quality = {quality}"
        )
        return -1.0  # Penalize trials that fail to produce clusters
    logger.info(
        f"Trial {trial.number}: Average intra‑cluster correlation = {quality:.4f}"
    )
    return quality


def run_dbscan_decorrelation_study(
    returns_df: pd.DataFrame, min_samples: int = 2, n_trials: int = 50
) -> dict:
    """
    Run an Optuna study to optimize DBSCAN’s eps so that clusters are as tight as possible.
    A tight (high correlation) cluster means that by selecting the top performer from each
    cluster, you end up with a portfolio of decorrelated assets.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and asset returns as columns.
        min_samples (int): Minimum samples for DBSCAN.
        n_trials (int): Number of trials to run.

    Returns:
        dict: Best parameters found, including the optimal eps.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_dbscan_decorrelation(trial, returns_df, min_samples),
        n_trials=n_trials,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    best_params = study.best_trial.params
    logger.info(
        f"Best eps: {best_params['eps']:.4f} with quality {study.best_value:.4f}"
    )
    return best_params
