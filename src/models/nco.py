import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.optimize import minimize

from utils.logger import logger


def cov_to_corr(cov):
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # Numerical stability
    return corr


def empirical_lpm(portfolio_returns, target=0, order=3):
    """
    Compute the empirical lower partial moment (LPM) of a return series.

    Parameters:
        portfolio_returns : array-like, historical portfolio returns.
        target          : target return level.
        order           : order of the LPM (default is 3).

    Returns:
        The LPM of the specified order.
    """
    diff = np.maximum(target - portfolio_returns, 0)
    return np.mean(diff**order)


def optimize_weights_objective(
    cov,
    mu=None,
    returns=None,
    objective="min_variance",  # Options: min_variance, kappa, blend, sharpe
    order=3,
    target=0,
    max_weight=1.0,
    allow_short=False,
    target_sum=1.0,
):
    """
    Optimize portfolio weights using a unified interface that supports several objectives.
    For min_variance and sharpe, expected returns (mu) and covariance (cov) are used.
    For kappa or half_kappa_sharpe, historical returns (returns) are also required.
    """
    n = cov.shape[0]
    lower_bound = -max_weight if allow_short else 0.0
    bounds = [(lower_bound, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - target_sum}

    if objective == "min_variance":
        if mu is None:

            def obj(w):
                return w.T @ cov @ w

        else:
            # Use a return-to-risk ratio (negative because we minimize)
            def obj(w):
                port_return = w @ mu
                port_var = w.T @ cov @ w
                return -port_return / np.sqrt(port_var) if port_var > 0 else 1e6

    elif objective == "kappa":
        if returns is None:
            raise ValueError(
                "Historical returns must be provided for kappa optimization."
            )

        def obj(w):
            port_returns = returns.values @ w
            port_mean = np.mean(port_returns)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            if lpm < 1e-8:
                return 1e6
            kappa = (port_mean - target) / (lpm ** (1.0 / order))
            return -kappa

    elif objective == "blend":
        # A simple combined objective: 50% kappa + 50% sharpe.
        if returns is None or mu is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for Kappa and Sharpe optimization."
            )

        def obj(w):
            port_returns = returns.values @ w
            port_mean = np.mean(port_returns)
            port_vol = np.sqrt(w.T @ cov @ w)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            if lpm < 1e-8:
                kappa_val = -1e6
            else:
                kappa_val = (port_mean - target) / (lpm ** (1.0 / order))
            sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6
            combined = 0.5 * kappa_val + 0.5 * sharpe_val
            return -combined

    elif objective == "sharpe":
        if mu is None:
            raise ValueError(
                "Expected returns (mu) must be provided for Sharpe optimization."
            )

        def obj(w):
            port_return = w @ mu
            port_vol = np.sqrt(w.T @ cov @ w)
            return -port_return / port_vol if port_vol > 0 else 1e6

    elif objective == "aggro":
        if returns is None or mu is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for aggro optimization."
            )

        def obj(w):
            # Compute daily portfolio returns
            port_returns = returns.values @ w
            # Compute cumulative return as the compounded product of (1 + daily_return)
            cumulative_return = np.prod(1 + port_returns) - 1
            # Compute risk-adjusted return using Sharpe (annualized or daily, depending on your data)
            port_mean = np.mean(port_returns)
            port_vol = np.sqrt(w.T @ cov @ w)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            if lpm < 1e-8:
                kappa_val = -1e6
            else:
                kappa_val = (port_mean - target) / (lpm ** (1.0 / order))
            sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6

            # Blend the two metrics. You may need to adjust the weighting factors.
            combined = (
                (1 / 3) * cumulative_return + (1 / 3) * sharpe_val + (1 / 3) * kappa_val
            )
            return -combined  # negative because we minimize

    else:
        print(
            "Unknown objective specified: {}. Defaulting to Sharpe optimal".format(
                objective
            )
        )

        def obj(w):
            port_return = w @ mu
            port_vol = np.sqrt(w.T @ cov @ w)
            return -port_return / port_vol if port_vol > 0 else 1e6

    init_weights = np.ones(n) / n
    result = minimize(
        obj, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    return result.x


def cluster_kmeans(corr: np.ndarray, max_clusters: int = 10) -> np.ndarray:
    """
    Cluster assets using KMeans on the correlation matrix.
    """
    # Transform correlation to a distance metric
    dist = np.sqrt(0.5 * (1 - corr))
    n_samples = dist.shape[0]

    max_valid_clusters = min(max_clusters, n_samples - 1) if n_samples > 1 else 1
    best_silhouette = -1.0
    best_labels = None

    for k in range(2, max_valid_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(dist)
        if len(np.unique(labels)) < 2:
            continue
        silhouette = silhouette_samples(dist, labels).mean()
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=int)

    return best_labels


def nested_clustered_optimization(
    cov: pd.DataFrame,
    mu: pd.Series = None,
    returns: pd.DataFrame = None,
    objective: str = "min_variance",  # Choose: min_variance, sharpe, kappa, blend, aggro
    max_clusters: int = 10,
    max_weight: float = 1.0,
    allow_short: bool = False,
    target: float = 0,
    order: int = 3,
    target_sum: float = 1.0,
) -> pd.Series:
    """
    Perform Nested Clustered Optimization with a flexible objective.
    For objectives requiring historical returns (kappa or blend),
    a 'returns' DataFrame must be provided.
    """
    # Filter assets with enough historical data
    min_data_threshold = cov.shape[0] * 0.5  # At least 50% valid history
    valid_assets = cov.index[cov.notna().sum(axis=1) >= min_data_threshold]

    if len(valid_assets) < 2:
        logger.warning(
            "Not enough valid assets after filtering. Skipping optimization."
        )
        return pd.Series(dtype=float)  # Return an empty portfolio

    cov = cov.loc[valid_assets, valid_assets]
    if mu is not None:
        mu = mu.loc[valid_assets]
    if returns is not None:
        returns = returns[valid_assets]

    # Create the correlation matrix and cluster the assets
    corr = cov_to_corr(cov)
    labels = cluster_kmeans(corr, max_clusters)
    unique_clusters = np.unique(labels)

    # Intra-cluster optimization: optimize weights for assets within each cluster
    intra_weights = pd.DataFrame(
        0, index=cov.index, columns=unique_clusters, dtype=float
    )
    for cluster in unique_clusters:
        cluster_assets = cov.index[labels == cluster]
        cluster_cov = cov.loc[cluster_assets, cluster_assets]
        cluster_mu = None if mu is None else mu.loc[cluster_assets]
        cluster_returns = None if returns is None else returns[cluster_assets]

        weights = optimize_weights_objective(
            cluster_cov,
            mu=cluster_mu,
            returns=cluster_returns,
            objective=objective,
            order=order,
            target=target,
            max_weight=max_weight,
            allow_short=allow_short,
            target_sum=target_sum,
        )
        intra_weights.loc[cluster_assets, cluster] = weights

    # Inter-cluster optimization: optimize how to weight each cluster
    reduced_cov = intra_weights.T @ cov @ intra_weights
    reduced_mu = None if mu is None else intra_weights.T @ mu

    # For historical returns aggregation, a simple (averaged) approach is used.
    reduced_returns = None
    if returns is not None:
        # Compute a mean return per cluster using intra-cluster weights
        reduced_returns = pd.Series(
            {
                cluster: (
                    returns.loc[:, intra_weights.index]
                    .mul(intra_weights[cluster], axis=1)
                    .sum(axis=1)
                    .mean()
                )
                for cluster in unique_clusters
            }
        )

    inter_weights = pd.Series(
        optimize_weights_objective(
            reduced_cov,
            mu=reduced_mu,
            returns=reduced_returns,
            objective=objective,
            order=order,
            target=target,
            max_weight=max_weight,
            allow_short=allow_short,
            target_sum=target_sum,
        ),
        index=unique_clusters,
    )

    # Combine intra- and inter-cluster weights to get the final portfolio allocation
    final_weights = intra_weights.mul(inter_weights, axis=1).sum(axis=1)
    final_weights = final_weights[
        final_weights.abs() >= 0.01
    ]  # Apply a minimum weight threshold

    if not isinstance(final_weights, pd.Series):
        final_weights = pd.Series(final_weights, index=intra_weights.index)

    return final_weights
