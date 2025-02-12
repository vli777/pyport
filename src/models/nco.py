import sys
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
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
    cov: pd.DataFrame,
    mu: Optional[Union[pd.Series, np.ndarray]] = None,
    returns: Optional[pd.DataFrame] = None,
    objective: str = "sharpe",
    order: int = 3,
    target: float = 0,
    max_weight: float = 1.0,
    allow_short: bool = False,
    target_sum: float = 1.0,
) -> np.ndarray:
    """
    Optimize portfolio weights using a unified interface that supports several objectives.
    For sharpe, expected returns (mu) and covariance (cov) are used.
    For min_cvar, kappa, sk_blend, sko_blend, omega, and aggro, historical returns (returns) are also required.

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        mu (Optional[Union[pd.Series, np.ndarray]]): Expected returns.
        returns (Optional[pd.DataFrame]): Historical returns. Must be a DataFrame with shape (T, n).
        objective (str): Objective to optimize. Options include "min_cvar", "kappa", "sk_blend", "sharpe",
                         "sko_blend", "omega", "aggro".
        order (int): Order for downside risk metrics (default 3).
        target (float): Target return (default 0).
        max_weight (float): Maximum weight per asset (default 1.0).
        allow_short (bool): Allow short positions (default False).
        target_sum (float): Sum of weights (default 1.0).

    Returns:
        np.ndarray: Optimized weights.
    """
    n = cov.shape[0]
    lower_bound = -max_weight if allow_short else 0.0
    bounds = [(lower_bound, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - target_sum}

    if objective == "min_vol_tail":
        if returns is None:
            raise ValueError("Historical returns must be provided for min_vol_tail optimization.")

        def obj(w: np.ndarray) -> float:
            r_vals = returns.values
            # Ensure r_vals is 2D (T x n)
            if r_vals.ndim == 1:
                r_vals = r_vals.reshape(-1, 1)
            if r_vals.shape[1] != n:
                raise ValueError(f"Shape mismatch: returns has {r_vals.shape[1]} column(s), expected {n}")
            
            # Calculate portfolio returns as a time series
            port_returns = r_vals @ w  # Shape: (T,)
            port_returns = np.atleast_1d(port_returns)
            
            # Volatility component (standard deviation)
            vol = np.std(port_returns)
            
            # CVaR component (tail risk)
            port_losses = -port_returns  # Convert returns to losses
            sorted_losses = np.sort(port_losses)
            alpha = 0.05  # Tail probability (worst 5% losses)
            num_tail = max(1, int(np.ceil(alpha * len(sorted_losses))))
            tail_losses = sorted_losses[:num_tail]
            cvar = np.mean(tail_losses)
            
            # Penalty if tail risk is below break-even (i.e. cvar < 0)
            penalty = 0.0
            if cvar < 0:
                # You can adjust lambda_penalty to control the trade-off
                lambda_penalty = 1.0  
                penalty = lambda_penalty * (-cvar)
            
            # The overall objective is to minimize volatility plus the penalty.
            return vol + penalty

    
    if objective == "min_cvar":
        if returns is None:
            raise ValueError(
                "Historical returns must be provided for min CVaR optimization."
            )

        def obj(w: np.ndarray) -> float:
            r_vals = returns.values
            if r_vals.ndim == 1:
                r_vals = r_vals.reshape(-1, 1)
            if r_vals.shape[1] != n:
                raise ValueError(
                    f"Shape mismatch: returns has {r_vals.shape[1]} column(s), expected {n}"
                )
            port_returns = r_vals @ w
            port_returns = np.atleast_1d(port_returns)

            # CVaR component (tail risk minimization)
            port_losses = -port_returns
            sorted_losses = np.sort(port_losses)
            alpha = 0.05  # tail risk %
            num_tail = max(1, int(np.ceil(alpha * len(sorted_losses))))
            tail_losses = sorted_losses[:num_tail]
            cvar = np.mean(tail_losses)

            # Volatility component (standard deviation)
            vol = np.std(port_returns)

            # Weight the two components as needed (tweak lambda for your preference)
            lambda_vol = 1.0  # example weight for volatility penalty
            return cvar + lambda_vol * vol

        # pure min cvar
        # def obj(w: np.ndarray) -> float:
        #     r_vals = returns.values
        #     # Ensure r_vals is 2D (T x n)
        #     if r_vals.ndim == 1:
        #         r_vals = r_vals.reshape(-1, 1)
        #     if r_vals.shape[1] != n:
        #         raise ValueError(
        #             f"Shape mismatch: returns has {r_vals.shape[1]} column(s), expected {n}"
        #         )
        #     port_returns = r_vals @ w  # Shape: (T,)
        #     port_returns = np.atleast_1d(port_returns)
        #     port_losses = -port_returns  # Convert returns to losses
        #     sorted_losses = np.sort(port_losses)
        #     alpha = 0.05  # Tail probability (5% worst losses)
        #     num_tail = max(1, int(np.ceil(alpha * len(sorted_losses))))
        #     tail_losses = sorted_losses[:num_tail]
        #     cvar = np.mean(tail_losses)
        #     return cvar
    elif objective == "min_var":
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

        def obj(w: np.ndarray) -> float:
            port_returns = returns.values @ w
            port_mean = np.mean(port_returns)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            if lpm < 1e-8:
                return 1e6
            kappa = (port_mean - target) / (lpm ** (1.0 / order))
            return -kappa

    elif objective == "sk_blend":
        # A simple combined objective: 50% kappa + 50% sharpe.
        if returns is None or mu is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for Kappa and Sharpe optimization."
            )

        def obj(w: np.ndarray) -> float:
            port_returns = returns.values @ w
            port_mean = np.mean(port_returns)
            port_vol = np.sqrt(w.T @ cov @ w)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            kappa_val = (
                (port_mean - target) / (lpm ** (1.0 / order)) if lpm > 1e-8 else -1e6
            )
            sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6
            combined = 0.5 * kappa_val + 0.5 * sharpe_val
            return -combined

    elif objective == "sharpe":
        if mu is None:
            raise ValueError(
                "Expected returns (mu) must be provided for Sharpe optimization."
            )

        def obj(w: np.ndarray) -> float:
            port_return = w @ mu
            port_vol = np.sqrt(w.T @ cov @ w)
            return -port_return / port_vol if port_vol > 0 else 1e6

    elif objective == "sko_blend":
        if returns is None or mu is None:
            raise ValueError(
                "Historical returns and expected returns (mu) must be provided for the blend."
            )

        def obj(w: np.ndarray) -> float:
            r_vals = returns.values
            if r_vals.ndim == 1:
                r_vals = r_vals.reshape(-1, 1)
            if r_vals.shape[1] != n:
                raise ValueError(
                    f"Shape mismatch: returns has {r_vals.shape[1]} column(s), expected {n}"
                )
            port_returns = r_vals @ w
            port_returns = np.atleast_1d(port_returns)
            port_mean = np.mean(port_returns)
            port_vol = np.sqrt(w.T @ cov @ w)
            sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6
            lpm = empirical_lpm(port_returns, target=target, order=order)
            kappa_val = (
                (port_mean - target) / (lpm ** (1.0 / order)) if lpm > 1e-8 else -1e6
            )
            threshold = target
            gain = (
                np.mean(port_returns[port_returns > threshold])
                if np.any(port_returns > threshold)
                else 1e-8
            )
            loss = (
                -np.mean(port_returns[port_returns < threshold])
                if np.any(port_returns < threshold)
                else 1e6
            )
            omega_val = gain / loss if loss > 1e-8 else -1e6
            combined = (1 / 3) * sharpe_val + (1 / 3) * kappa_val + (1 / 3) * omega_val
            return -combined

    elif objective == "omega":
        if returns is None:
            raise ValueError(
                "Historical returns must be provided for Omega optimization."
            )

        def obj(w: np.ndarray) -> float:
            r_vals = returns.values
            if r_vals.ndim == 1:
                r_vals = r_vals.reshape(-1, 1)
            if r_vals.shape[1] != n:
                raise ValueError(
                    f"Shape mismatch: returns has {r_vals.shape[1]} column(s), expected {n}"
                )
            port_returns = r_vals @ w
            port_returns = np.atleast_1d(port_returns)
            threshold = target
            gain = (
                np.mean(port_returns[port_returns > threshold])
                if np.any(port_returns > threshold)
                else 1e-8
            )
            loss = (
                -np.mean(port_returns[port_returns < threshold])
                if np.any(port_returns < threshold)
                else 1e6
            )
            omega_ratio_val = gain / loss if loss > 1e-8 else -1e6
            return -omega_ratio_val

    elif objective == "aggro":
        if returns is None or mu is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for aggro optimization."
            )

        def obj(w: np.ndarray) -> float:
            port_returns = returns.values @ w
            cumulative_return = np.prod(1 + port_returns) - 1
            port_mean = np.mean(port_returns)
            port_vol = np.sqrt(w.T @ cov @ w)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            kappa_val = (
                (port_mean - target) / (lpm ** (1.0 / order)) if lpm > 1e-8 else -1e6
            )
            sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6
            combined = (
                (1 / 3) * cumulative_return + (1 / 3) * sharpe_val + (1 / 3) * kappa_val
            )
            return -combined

    else:
        print(
            f"Unknown objective specified: {objective}. Defaulting to Sharpe optimal."
        )

        def obj(w: np.ndarray) -> float:
            port_return = w @ mu
            port_vol = np.sqrt(w.T @ cov @ w)
            return -port_return / port_vol if port_vol > 0 else 1e6

    init_weights = np.ones(n) / n
    feasible_min = n * lower_bound
    feasible_max = n * max_weight
    if not (feasible_min <= target_sum <= feasible_max):
        raise ValueError(
            f"Infeasible target_sum: {target_sum}. It must be between {feasible_min} and {feasible_max} for n={n} assets."
        )

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
    mu: Optional[pd.Series] = None,
    returns: Optional[pd.DataFrame] = None,
    objective: str = "sharpe",
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

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        mu (Optional[pd.Series]): Expected returns.
        returns (Optional[pd.DataFrame]): Historical returns (time series) with assets as columns.
        objective (str): Optimization objective.
        max_clusters (int): Maximum number of clusters.
        max_weight (float): Maximum weight per asset.
        allow_short (bool): Allow short positions.
        target (float): Target return (default 0).
        order (int): Order for downside risk metrics.
        target_sum (float): Sum of weights (default 1.0).

    Returns:
        pd.Series: Final portfolio weights.
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

    # For historical returns aggregation, build a DataFrame where each column is the cluster's time series.
    reduced_returns = None
    if returns is not None:
        reduced_returns = pd.DataFrame(
            {
                cluster: (
                    returns.loc[:, intra_weights.index]
                    .mul(intra_weights[cluster], axis=1)
                    .sum(axis=1)
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
