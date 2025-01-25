import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from scipy.optimize import minimize

from utils import logger


def cov_to_corr(cov):
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # Numerical stability
    return corr


def optimize_weights(cov, mu=None, max_weight=1.0, allow_short=False):
    """
    Compute portfolio weights using inverse covariance or minimize variance
    with per-asset constraints.
    """
    n = cov.shape[0]

    if mu is None:

        def objective(w):
            return w.T @ cov @ w

    else:

        def objective(w):
            portfolio_return = w @ mu
            portfolio_variance = w.T @ cov @ w
            return -portfolio_return / np.sqrt(portfolio_variance)

    init_weights = np.ones(n) / n
    lower_bound = -max_weight if allow_short else 0.0
    bounds = [(lower_bound, max_weight) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    return result.x


def optimize_weights_sharpe(cov, mu, max_weight=1.0, allow_short=False, target_sum=1.0):
    """
    Compute portfolio weights that maximize the Sharpe ratio.

    Args:
        cov (np.ndarray): Covariance matrix of asset returns.
        mu (np.ndarray): Expected returns of assets.
        max_weight (float): Maximum allowed weight per asset (between 0 and 1).

    Returns:
        np.ndarray: Optimal weights that maximize the Sharpe ratio.
    """
    # Number of assets
    n = cov.shape[0]

    # Define the objective function (negative Sharpe ratio)
    def negative_sharpe(weights):
        portfolio_return = weights @ mu
        portfolio_volatility = np.sqrt(weights.T @ cov @ weights)
        return -portfolio_return / portfolio_volatility

    # Initial guess: Equal weights
    init_weights = np.clip(np.ones(n) / n, -max_weight, max_weight)

    # Choose lower bound based on allow_short
    bounds = [
        (-max_weight, max_weight) if allow_short else (0, max_weight) for _ in range(n)
    ]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - target_sum}

    # Optimize
    result = minimize(
        negative_sharpe,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        logger.error(f"Optimization failed: {result.message}")
        logger.debug(f"Initial weights: {init_weights}")
        logger.debug(f"Bounds: {bounds}")
        logger.debug(f"Constraints: {constraints}")
        logger.debug(f"Covariance matrix: {cov}")
        logger.debug(f"Expected returns: {mu}")
        raise ValueError("Optimization failed:", result.message)

    return result.x


def cluster_kmeans(corr: np.ndarray, max_clusters: int = 10) -> np.ndarray:
    """
    Cluster assets using KMeans on the correlation matrix.

    Args:
        corr (np.ndarray): Correlation matrix.
        max_clusters (int): Maximum number of clusters to test.

    Returns:
        np.ndarray: Array of cluster labels for each asset.
    """
    # Compute the distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr))
    n_samples = dist.shape[0]

    # Determine the maximum allowable clusters to prevent silhouette errors
    max_valid_clusters = min(max_clusters, n_samples - 1) if n_samples > 1 else 1

    best_silhouette = -1.0
    best_labels = None

    # Loop over the valid range of cluster counts
    for k in range(2, max_valid_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(dist)

        # Ensure that at least two unique labels exist before computing silhouette
        if len(np.unique(labels)) < 2:
            continue

        # Compute the mean silhouette score for this clustering
        silhouette = silhouette_samples(dist, labels).mean()
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_labels = labels

    # If no valid clustering was found, default to a single cluster for all assets
    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=int)

    return best_labels


def nested_clustered_optimization(
    cov: pd.DataFrame,
    mu: pd.Series = None,
    max_clusters: int = 10,
    sharpe: bool = False,
    max_weight: float = 1.0,
    allow_short: bool = False,
) -> pd.Series:
    """
    Implement Nested Clustered Optimization with support for short positions.
    """
    # Correlation matrix and clustering
    corr = cov_to_corr(cov)
    labels = cluster_kmeans(corr, max_clusters)
    unique_clusters = np.unique(labels)

    # Intra-cluster optimization
    intra_weights = pd.DataFrame(
        0, index=cov.index, columns=unique_clusters, dtype=float
    )
    for cluster in unique_clusters:
        cluster_assets = cov.index[labels == cluster]
        cluster_cov = cov.loc[cluster_assets, cluster_assets].values
        cluster_mu = None if mu is None else mu.loc[cluster_assets].values

        # Pass `allow_short` to the optimization functions
        if sharpe:
            intra_weights.loc[cluster_assets, cluster] = optimize_weights_sharpe(
                cluster_cov, cluster_mu, max_weight=max_weight, allow_short=allow_short
            )
        else:
            intra_weights.loc[cluster_assets, cluster] = optimize_weights(
                cluster_cov, cluster_mu, max_weight=max_weight, allow_short=allow_short
            )

    # Inter-cluster optimization
    reduced_cov = intra_weights.T @ cov @ intra_weights
    reduced_mu = None if mu is None else intra_weights.T @ mu

    inter_weights = (
        pd.Series(
            optimize_weights_sharpe(
                reduced_cov, reduced_mu, max_weight=max_weight, allow_short=allow_short
            ),
            index=unique_clusters,
        )
        if sharpe
        else pd.Series(
            optimize_weights(
                reduced_cov, reduced_mu, max_weight=max_weight, allow_short=allow_short
            ),
            index=unique_clusters,
        )
    )

    # Combine intra- and inter-cluster weights
    final_weights = intra_weights.mul(inter_weights, axis=1).sum(axis=1)

    # Apply a minimum weight threshold (preserves sign for shorts)
    final_weights = final_weights[final_weights.abs() >= 0.01]

    return final_weights
