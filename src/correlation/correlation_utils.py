import pandas as pd
import numpy as np
from typing import List
from sklearn.covariance import LedoitWolf

from utils.logger import logger


def compute_lw_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Ledoit-Wolf shrinkage covariance matrix for asset returns.

    Args:
        df (pd.DataFrame): Asset returns DataFrame (T x n), where T is time.

    Returns:
        pd.DataFrame: Ledoit-Wolf covariance matrix (n x n).
    """
    lw = LedoitWolf()
    covariance = lw.fit(df).covariance_
    return pd.DataFrame(covariance, index=df.columns, columns=df.columns)


def compute_lw_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a Ledoit-Wolf covariance and converts it to a correlation matrix.
    """
    covariance = compute_lw_covariance(df)
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


def compute_distance_matrix(
    returns_df: pd.DataFrame, scale_distances: bool = False
) -> np.ndarray:
    """
    Compute a normalized distance matrix from the correlation matrix of returns.
    The conversion uses the transformation: distance = (1 - correlation) / 2.
    Optionally, if scale_distances is True, the distance matrix is re-scaled via
    min–max normalization to fully span the [0, 1] interval.
    This function ensures the returned distance matrix is a NumPy array to prevent
    indexing issues later on.
    """
    # Compute the correlation matrix (user-defined function).
    corr_matrix = compute_correlation_matrix(returns_df)
    # Ensure the correlation matrix is a NumPy array.
    corr_matrix = np.asarray(corr_matrix)
    # Convert correlations to distances in [0, 1].
    distance_matrix = (1 - corr_matrix) / 2

    if scale_distances:
        dmin, dmax = np.min(distance_matrix), np.max(distance_matrix)
        if not np.isclose(dmax, dmin):
            distance_matrix = (distance_matrix - dmin) / (dmax - dmin)
            logger.debug("Distance matrix re-scaled to full [0, 1] range.")
    return distance_matrix
