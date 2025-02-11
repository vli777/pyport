import hashlib
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import List
from sklearn.covariance import LedoitWolf

from utils.performance_metrics import kappa_ratio, sharpe_ratio


def validate_matrix(matrix, matrix_name: str):
    """
    Validate that a matrix does not contain NaN or infinite values.

    Args:
        matrix: The matrix to validate.
        matrix_name (str): Name of the matrix (used for error messages).
    """
    if matrix.isnull().values.any():
        raise ValueError(f"{matrix_name} contains NaN values.")
    if np.isinf(matrix.values).any():
        raise ValueError(f"{matrix_name} contains infinite values.")


def calculate_condensed_distance_matrix(corr_matrix):
    """
    Calculate the condensed distance matrix from a correlation matrix.

    Args:
        corr_matrix: Correlation matrix.

    Returns:
        np.ndarray: Condensed distance matrix.
    """
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix.values, 0)
    validate_matrix(distance_matrix, "Distance matrix")
    return squareform(distance_matrix)


def hierarchical_clustering(
    corr_matrix: pd.DataFrame,
    distance_threshold: float,
    linkage_method: str,
    plot: bool,
) -> List[int]:
    """
    Perform hierarchical clustering and return cluster assignments.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        distance_threshold (float): Threshold for forming clusters.
        linkage_method (str): Method for hierarchical clustering.
        plot (bool): Whether to plot the clustering dendrogram.

    Returns:
        List[int]: Cluster assignments for each item.
    """
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix.values, 0)

    # Ensure no NaNs/Infs exist
    validate_matrix(distance_matrix, "Distance matrix")

    # Convert to condensed format for clustering
    condensed_distance_matrix = squareform(distance_matrix)

    # Perform hierarchical clustering
    linked = linkage(condensed_distance_matrix, method=linkage_method)

    if plot:
        fig = ff.create_dendrogram(
            linked, labels=corr_matrix.index.tolist(), linkagefun=lambda x: linked
        )
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=0,
                x1=len(corr_matrix),
                y0=distance_threshold,
                y1=distance_threshold,
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="Hierarchical Clustering Dendrogram",
            xaxis_title="Ticker",
            yaxis_title="Distance (1 - Correlation)",
        )
        fig.show()

    return fcluster(linked, t=distance_threshold, criterion="distance")


def compute_lw_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a Ledoit-Wolf covariance and converts it to a correlation matrix.
    """
    lw = LedoitWolf()
    covariance = lw.fit(df).covariance_
    stddev = np.sqrt(np.diag(covariance))
    corr_matrix = covariance / np.outer(stddev, stddev)
    np.fill_diagonal(corr_matrix, 0)
    return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)


def compute_correlation_matrix(
    df: pd.DataFrame, lw_threshold: int = 50, use_abs: bool = True
) -> pd.DataFrame:
    """
    Compute a correlation matrix for the given returns DataFrame.
    Uses the Ledoit-Wolf estimator if the number of columns exceeds lw_threshold.
    Otherwise, uses the standard Pearson correlation.

    Args:
        df (pd.DataFrame): DataFrame of returns with tickers as columns.
        lw_threshold (int): If number of columns > lw_threshold, use Ledoit-Wolf.
        use_abs (bool): If True, take the absolute value of correlations.

    Returns:
        pd.DataFrame: Correlation matrix with the diagonal filled with zeros.
    """
    if len(df.columns) > lw_threshold:
        # Use the Ledoit-Wolf based correlation matrix.
        corr_matrix = compute_lw_correlation(df)
    else:
        # Use standard correlation.
        corr_matrix = df.corr()
        if use_abs:
            corr_matrix = corr_matrix.abs()
        # Fill diagonal with zeros.
        np.fill_diagonal(corr_matrix.values, 0)
    return corr_matrix


def compute_performance_metrics(
    returns_df: pd.DataFrame,
    risk_free_rate: float,
    objective_weights={
        "sharpe": 1.0,  # Default: Optimize purely for Sharpe Ratio
    },
) -> pd.Series:

    metrics = {}

    for ticker in returns_df.columns:
        ticker_returns = returns_df[ticker].dropna()
        if ticker_returns.empty:
            metrics[ticker] = np.nan  # Return NaN instead of skipping
            continue

        # Compute individual metrics
        cumulative_return = (ticker_returns + 1).prod() - 1  # Simple return formula
        sr = sharpe_ratio(ticker_returns, risk_free_rate)
        kp = kappa_ratio(ticker_returns, order=3) if "kappa" in objective_weights else 0
        volatility = ticker_returns.std() if "min_variance" in objective_weights else 0

        # Apply penalty for negative cumulative return
        penalty = 0
        if cumulative_return < 0:
            penalty = 2 * abs(
                cumulative_return
            )  # Increase penalty multiplier if needed

        # Weighted composite score
        composite = (
            objective_weights.get("sharpe", 1.0) * sr
            + objective_weights.get("kappa", 0.0)
            * kp  # Included if kappa is part of the objective
            - objective_weights.get("min_variance", 0.0)
            * volatility  # Included if min_variance is used
            + objective_weights.get("cumulative_return", 0.0)
            * cumulative_return  # Included if cumulative return is used
            - penalty  # Apply additional penalty for negative returns
        )

        metrics[ticker] = composite

    return pd.Series(metrics)


def get_objective_weights(objective: str) -> dict:
    """
    Returns the objective weight dictionary based on the given objective.

    Args:
        objective (str): The optimization objective. Choices: ["minvar", "sharpe", "blend", "aggro"]

    Returns:
        dict: Objective weight dictionary for computing performance metrics.
    """
    objective_mappings = {
        "minvar": {
            "cumulative_return": 0.0,  # Ignore total return
            "sharpe": 0.0,  # Ignore Sharpe ratio
            "kappa": 0.0,  # Ignore Kappa ratio
            "min_variance": 1.0,  # Fully minimize variance
        },
        "kappa": {
            "cumulative_return": 0.0,  # Ignore total return
            "sharpe": 0.0,  # Ignore Sharpe ratio
            "kappa": 1.0,  # Full maximize Kappa ratio
            "min_variance": 0.0,  # Ignore variance
        },
        "blend": {
            "cumulative_return": 0.0,  # Ignore total return
            "sharpe": 0.5,  # Some risk-adjusted return
            "kappa": 0.5,  # Some tail-risk adjustment
            "min_variance": 0.0,  # Ignore variance minimization
        },
        "sharpe": {
            "cumulative_return": 0.0,  # Ignore total return
            "sharpe": 1.0,  # Fully maximize Sharpe ratio
            "kappa": 0.0,  # Ignore Kappa
            "min_variance": 0.0,  # Ignore variance
        },
        "aggro": {
            "cumulative_return": 1 / 3,  # Prioritize returns
            "sharpe": 1 / 3,  # Some risk-adjusted return
            "kappa": 1 / 3,  # Some tail-risk adjustment
            "min_variance": 0.0,  # Ignore variance minimization
        },
    }

    if objective not in objective_mappings:
        raise ValueError(
            f"Unknown objective: {objective}. Choose from {list(objective_mappings.keys())}"
        )

    return objective_mappings[objective]


def volatility_constraint(w, cov, max_vol=0.2):
    """Ensures portfolio volatility does not exceed a given threshold."""
    return max_vol - np.sqrt(w.T @ cov @ w)  # Must be >= 0
