from collections import defaultdict
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

from correlation.correlation_utils import (
    validate_matrix,
    calculate_condensed_distance_matrix,
    hierarchical_clustering,
)
from utils import logger


def filter_correlated_groups(
    returns_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    correlation_threshold: float = 0.8,
    sharpe_threshold: float = 0.005,
    linkage_method: str = "average",
    top_n: int = None,
    plot: bool = False,
) -> List[str]:
    """
    Iteratively filter correlated tickers based on correlation and Sharpe Ratio.
    Handles stocks with different historical lengths correctly.
    """
    total_excluded: set = set()
    iteration = 1

    if top_n is None:
        group_size = len(returns_df.columns)
        top_n = max(1, int(group_size * 0.1))

    while len(returns_df.columns) > top_n:
        # Align stocks to their **common overlapping window**
        min_common_date = returns_df.dropna(how="all").index.min()
        aligned_df = returns_df.loc[min_common_date:]

        # Use Ledoit-Wolf shrinkage if >50 assets, else standard correlation
        if len(aligned_df.columns) > 50:
            lw = LedoitWolf()
            covariance_matrix = lw.fit(aligned_df).covariance_
            stddev = np.sqrt(np.diag(covariance_matrix))
            corr_matrix = covariance_matrix / np.outer(stddev, stddev)
            np.fill_diagonal(corr_matrix, 0)
            corr_matrix = pd.DataFrame(
                corr_matrix, index=aligned_df.columns, columns=aligned_df.columns
            )
        else:
            corr_matrix = aligned_df.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)

        # Validate the correlation matrix
        validate_matrix(corr_matrix, "Correlation matrix")

        # Convert correlation to distance
        condensed_distance_matrix = calculate_condensed_distance_matrix(corr_matrix)
        distance_threshold = 1 - correlation_threshold

        cluster_assignments = hierarchical_clustering(
            corr_matrix=corr_matrix,
            condensed_distance_matrix=condensed_distance_matrix,
            distance_threshold=distance_threshold,
            linkage_method=linkage_method,
            plot=plot,
        )

        # Group tickers into clusters
        clusters = defaultdict(list)
        for ticker, cluster_label in zip(aligned_df.columns, cluster_assignments):
            clusters[cluster_label].append(ticker)

        correlated_groups = [
            set(group) for group in clusters.values() if len(group) > 1
        ]

        if not correlated_groups:
            break  # No more correlated groups

        # Pass min history check into `select_best_tickers`
        excluded_tickers = select_best_tickers(
            performance_df=performance_df,
            correlated_groups=correlated_groups,
            sharpe_threshold=sharpe_threshold,
            top_n=top_n,
            min_history=len(aligned_df) * 0.5,  # Ensure at least 50% available data
        )

        if not excluded_tickers:
            break  # No more tickers to exclude

        total_excluded.update(excluded_tickers)
        returns_df = returns_df.drop(columns=excluded_tickers)

        iteration += 1

    filtered_symbols = returns_df.columns.tolist()

    logger.info(f"{len(total_excluded)} tickers excluded")
    logger.info(f"{len(filtered_symbols)} De-correlated tickers: {filtered_symbols}")

    return filtered_symbols


def select_best_tickers(
    performance_df: pd.DataFrame,
    correlated_groups: list,
    sharpe_threshold: float = 0.005,
    top_n: Optional[int] = None,
    min_history: int = None,
) -> set:
    """
    Select top N tickers from each correlated group based on Sharpe Ratio, Total Return, and history length.

    Args:
        performance_df (pd.DataFrame): DataFrame with performance metrics.
        correlated_groups (list of sets): Groups of highly correlated tickers.
        sharpe_threshold (float): Threshold for considering Sharpe ratios similar.
        min_history (int, optional): Minimum number of observations required.

    Returns:
        set: Tickers to exclude.
    """
    tickers_to_exclude = set()

    for group in correlated_groups:
        if len(group) < 2:
            continue

        # Get performance metrics for the group
        group_metrics = performance_df.loc[list(group)].copy()

        # Enforce a minimum history length
        if min_history:
            valid_tickers = group_metrics.index[
                performance_df["History Length"] >= min_history
            ]
            group_metrics = group_metrics.loc[valid_tickers]

        if group_metrics.empty:
            continue  # Skip empty groups

        max_sharpe = group_metrics["Sharpe Ratio"].max()

        # Select tickers with Sharpe within the threshold of max Sharpe
        top_candidates = group_metrics[
            group_metrics["Sharpe Ratio"] >= (max_sharpe - sharpe_threshold)
        ]

        # Determine number of tickers to keep
        if top_n is None:
            group_size = len(group)
            dynamic_n = max(1, int(group_size * 0.1))
            current_top_n = min(dynamic_n, len(top_candidates))
        else:
            current_top_n = min(top_n, len(top_candidates))

        # Select top tickers based on Total Return
        top_n_candidates = top_candidates.nlargest(
            current_top_n, "Total Return"
        ).index.tolist()

        # Remove tickers not in the selected top N
        to_keep = set(top_n_candidates)
        to_exclude = group - to_keep
        tickers_to_exclude.update(to_exclude)

    return tickers_to_exclude
