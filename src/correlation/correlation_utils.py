import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import List
from sklearn.covariance import LedoitWolf


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
