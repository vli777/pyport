import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


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
    condensed_distance_matrix: np.ndarray,
    distance_threshold: float,
    linkage_method: str,
    plot: bool,
) -> List[int]:
    """
    Perform hierarchical clustering and return cluster assignments.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        condensed_distance_matrix (np.ndarray): Condensed distance matrix.
        distance_threshold (float): Threshold for forming clusters.
        linkage_method (str): Method for hierarchical clustering.
        plot (bool): Whether to plot the clustering dendrogram

    Returns:
        List[int]: Cluster assignments for each item.
    """
    linked = linkage(condensed_distance_matrix, method=linkage_method)

    if plot:
        # Create a clustermap with Seaborn
        sns.clustermap(
            corr_matrix,
            row_cluster=True,
            col_cluster=True,
            method=linkage_method,
            cmap="vlag",  # Diverging colormap
            linewidths=0.5,
            figsize=(12, 10),
            dendrogram_ratio=(0.2, 0.2),  # Adjust dendrogram size
        )
        plt.axhline(
            y=distance_threshold,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label="Distance Threshold",
        )
        plt.legend(loc="upper right")
        plt.title("Hierarchical Clustering Dendrogram")
        plt.show()

    return fcluster(linked, t=distance_threshold, criterion="distance")
