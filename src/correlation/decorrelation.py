from collections import defaultdict
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

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
    plot: bool = False,
) -> List[str]:
    """
    Iteratively filter correlated tickers based on a specified correlation threshold and Sharpe Ratio.
    """
    total_excluded: set = set()
    iteration = 1

    while True:
        if len(returns_df.columns) < 2:
            logger.info("Less than two tickers remain. Stopping iteration.")
            break

        corr_matrix = returns_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
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

        # Group tickers
        clusters = defaultdict(list)
        for ticker, cluster_label in zip(returns_df.columns, cluster_assignments):
            clusters[cluster_label].append(ticker)

        # Identify correlated groups (clusters with more than one ticker)
        correlated_groups = [
            set(group) for group in clusters.values() if len(group) > 1
        ]

        if not correlated_groups:
            logger.info("No correlated groups found. Stopping iteration.")
            break

        # Select tickers to exclude based on Sharpe Ratio
        excluded_tickers = select_best_tickers(
            performance_df=performance_df,
            correlated_groups=correlated_groups,
            sharpe_threshold=sharpe_threshold,
        )

        if not excluded_tickers:
            logger.info("No more tickers to exclude. Stopping iteration.")
            break

        total_excluded.update(excluded_tickers)
        logger.info(f"Iteration {iteration}: Excluded tickers: {excluded_tickers}")

        # Drop excluded tickers from the returns DataFrame
        returns_df = returns_df.drop(columns=excluded_tickers)

        iteration += 1

    logger.info(f"Total excluded tickers: {total_excluded}")
    return returns_df.columns.tolist()


def select_best_tickers(
    performance_df: pd.DataFrame,
    correlated_groups: list,
    sharpe_threshold: float = 0.005,
) -> set:
    """
    Select top N tickers from each correlated group based on Sharpe Ratio and Total Return.

    Args:
        performance_df (pd.DataFrame): DataFrame with performance metrics.
        correlated_groups (list of sets): Groups of highly correlated tickers.
        sharpe_threshold (float): Threshold to consider Sharpe ratios as similar.

    Returns:
        set: Tickers to exclude.
    """
    tickers_to_exclude = set()

    for group in correlated_groups:
        if len(group) < 2:
            continue

        logger.info(f"Evaluating group of correlated tickers: {group}")
        group_metrics = performance_df.loc[list(group)]
        max_sharpe = group_metrics["Sharpe Ratio"].max()
        logger.info(f"Maximum Sharpe Ratio in group: {max_sharpe:.4f}")

        # Identify tickers within the Sharpe threshold of the max
        top_candidates = group_metrics[
            group_metrics["Sharpe Ratio"] >= (max_sharpe - sharpe_threshold)
        ]

        # Dynamically determine how many tickers to select: top 10% of the group
        group_size = len(group)
        dynamic_n = max(1, int(group_size * 0.1))

        # Adjust dynamic_n based on available candidates
        dynamic_n = min(dynamic_n, len(top_candidates))

        # Select top 'dynamic_n' based on Total Return among the candidates
        top_n = top_candidates.nlargest(dynamic_n, "Total Return").index.tolist()
        logger.info(f"Selected top {dynamic_n} tickers: {top_n} from group {group}")

        # Exclude other tickers in the group
        to_keep = set(top_n)
        to_exclude = group - to_keep
        tickers_to_exclude.update(to_exclude)

    return tickers_to_exclude
