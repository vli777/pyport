from collections import defaultdict
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np


from correlation.correlation_utils import (
    compute_lw_correlation,
    hierarchical_clustering,
)
from correlation.optimize_correlation import optimize_correlation_threshold
from utils.caching_utils import load_thresholds_from_pickle, save_thresholds_to_pickle
from utils import logger


def filter_correlated_groups(
    returns_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    correlation_threshold: Optional[float] = None,
    sharpe_threshold: float = 0.005,
    linkage_method: str = "average",
    top_n: Optional[int] = None,
    plot: bool = False,
    n_jobs: int = -1,
    cache_filename: str = "optuna_cache/correlation_thresholds.pkl",
    reoptimize: bool = False,
) -> List[str]:
    """
    Iteratively filters correlated tickers based on correlation and Sharpe Ratio.
    Allows caching of an optimized correlation threshold to avoid re-computation.
    """

    # --------------------------------------------------------------------------
    # 1. Handle correlation threshold caching / re-optimization logic
    # --------------------------------------------------------------------------
    if correlation_threshold is None or reoptimize:
        # 1. Attempt to load thresholds from your pickle cache
        thresholds_dict = load_thresholds_from_pickle(cache_filename)

        # 2. If no cached value or we're forcing re-optimization, call the existing optimization function
        if not thresholds_dict or reoptimize:
            best_params, best_value = optimize_correlation_threshold(
                returns_df=returns_df,
                performance_df=performance_df,
                # add any other parameters needed by your optimizer:
                # market_returns=..., risk_free_rate=..., etc.
                sharpe_threshold=sharpe_threshold,
                linkage_method=linkage_method,
            )
            # Grab the new threshold from the optimization results
            correlation_threshold = best_params.get("correlation_threshold", 0.8)

            # 3. Update the cache and save to pickle
            thresholds_dict["correlation_threshold"] = correlation_threshold
            save_thresholds_to_pickle(thresholds_dict, cache_filename)
        else:
            # 4. If already cached and no re-opt, use the stored threshold
            correlation_threshold = thresholds_dict.get("correlation_threshold", 0.8)

    # --------------------------------------------------------------------------
    # 2. Begin correlation filtering with the final correlation_threshold
    # --------------------------------------------------------------------------
    total_excluded: Set[str] = set()

    # If no explicit top_n is given, default to 10% of total
    if top_n is None:
        group_size = len(returns_df.columns)
        top_n = max(1, int(group_size * 0.1))

    iteration = 1
    working_df = returns_df.copy()

    while len(working_df.columns) > top_n:
        # Align stocks to their common overlapping window
        min_common_date = working_df.dropna(how="all").index.min()
        aligned_df = working_df.loc[min_common_date:]

        # Compute correlation matrix (LedoitWolf if large)
        if len(aligned_df.columns) > 50:
            corr_matrix = compute_lw_correlation(aligned_df)
        else:
            corr_matrix = aligned_df.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)

        # Convert correlation -> distance = (1 - correlation)
        # Then do hierarchical clustering with the chosen linkage
        distance_threshold = 1 - correlation_threshold
        cluster_assignments = hierarchical_clustering(
            corr_matrix=corr_matrix,
            distance_threshold=distance_threshold,
            linkage_method=linkage_method,
            plot=plot,
            n_jobs=n_jobs,
        )

        # Group tickers by cluster
        clusters = defaultdict(list)
        for ticker, cluster_label in zip(aligned_df.columns, cluster_assignments):
            clusters[cluster_label].append(ticker)

        correlated_groups = [set(g) for g in clusters.values() if len(g) > 1]
        if not correlated_groups:
            logger.info("No more correlated groups found. Exiting.")
            break

        # Exclude lesser tickers from each correlated group
        excluded_tickers = select_best_tickers(
            performance_df=performance_df,
            correlated_groups=correlated_groups,
            sharpe_threshold=sharpe_threshold,
            top_n=top_n,
            min_history=len(aligned_df) * 0.5,
        )

        if not excluded_tickers:
            logger.info("No tickers to exclude. Exiting.")
            break

        total_excluded.update(excluded_tickers)
        working_df = working_df.drop(columns=excluded_tickers)
        iteration += 1

    # Final de-correlated symbols
    filtered_symbols = working_df.columns.tolist()
    logger.info(f"Correlation filter completed. Iterations: {iteration}")
    logger.info(f"{len(total_excluded)} tickers excluded.")
    logger.info(f"{len(filtered_symbols)} remaining tickers: {filtered_symbols}")

    return filtered_symbols


def select_best_tickers(
    performance_df: pd.DataFrame,
    correlated_groups: list,
    sharpe_threshold: float = 0.005,
    top_n: Optional[int] = None,
    min_history: int = None,
) -> Set[str]:
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

        group_metrics = performance_df.loc[list(group)].copy()

        # Enforce a minimum history length
        if min_history is not None and "History Length" in group_metrics.columns:
            valid_mask = group_metrics["History Length"] >= min_history
            group_metrics = group_metrics.loc[valid_mask]

        if group_metrics.empty:
            continue

        max_sharpe = group_metrics["Sharpe Ratio"].max()
        top_candidates = group_metrics[
            group_metrics["Sharpe Ratio"] >= (max_sharpe - sharpe_threshold)
        ]

        # If no explicit top_n, keep 10% of the group (but at least 1)
        if top_n is None:
            group_size = len(group)
            dynamic_n = max(1, int(group_size * 0.1))
            current_top_n = min(dynamic_n, len(top_candidates))
        else:
            current_top_n = min(top_n, len(top_candidates))

        # Sort the top candidates by total return desc
        top_n_candidates = top_candidates.nlargest(
            current_top_n, "Total Return"
        ).index.tolist()

        # Everything else gets excluded
        to_keep = set(top_n_candidates)
        to_exclude = group - to_keep
        tickers_to_exclude.update(to_exclude)

    return tickers_to_exclude
