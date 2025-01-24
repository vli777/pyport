import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import numpy as np

from utils import logger


def filter_correlated_groups(
    returns_df,
    performance_df,
    sharpe_threshold=0.005,
    correlation_threshold=0.8,
    linkage_method="average",
    plot=False,
):
    """
    Iteratively filter correlated tickers based on correlation and Sharpe Ratio.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns of tickers.
        performance_df (pd.DataFrame): DataFrame containing performance metrics (e.g., Sharpe Ratio) indexed by ticker.
        sharpe_threshold (float): Threshold to determine eligible tickers based on Sharpe Ratio.
        correlation_threshold (float): Threshold to determine if tickers are highly correlated.
        linkage_method (str): Method to use for hierarchical clustering.
        plot (bool): Whether to plot the dendrogram.

    Returns:
        List[str]: List of filtered ticker symbols.
    """
    total_excluded = set()
    iteration = 1

    while True:
        # Check if the number of tickers is less than 2
        if len(returns_df.columns) < 2:
            # print("Less than two tickers remain. Stopping iteration.")
            break

        # Compute the correlation matrix and set diagonal to zero
        corr_matrix = returns_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)

        # Debug: Verify that the diagonal is zero
        diag = np.diag(corr_matrix)
        # print(f"Iteration {iteration}: Diagonal of correlation matrix: {diag}")

        # Check for NaN or infinite values in correlation matrix
        if corr_matrix.isnull().values.any():
            raise ValueError("Correlation matrix contains NaN values.")
        if np.isinf(corr_matrix.values).any():
            raise ValueError("Correlation matrix contains infinite values.")

        # Compute distance matrix
        distance_matrix = 1 - corr_matrix

        # Ensure the diagonal is zero
        np.fill_diagonal(distance_matrix.values, 0)

        # Debug: Verify that the distance matrix diagonal is zero
        diag_dist = np.diag(distance_matrix)
        # print(f"Iteration {iteration}: Diagonal of distance matrix: {diag_dist}")

        # Check for NaN or infinite values in distance matrix
        if distance_matrix.isnull().values.any():
            raise ValueError("Distance matrix contains NaN values.")
        if np.isinf(distance_matrix.values).any():
            raise ValueError("Distance matrix contains infinite values.")

        # Convert to condensed distance matrix
        try:
            condensed_distance_matrix = squareform(distance_matrix)
        except ValueError as e:
            print("Error during squareform conversion:", e)
            print("Distance matrix:\n", distance_matrix)
            raise

        # Ensure correlation_threshold is within [0,1]
        if not (0 <= correlation_threshold <= 1):
            raise ValueError(
                f"correlation_threshold must be between 0 and 1, got {correlation_threshold}"
            )

        # Convert correlation threshold to distance threshold
        distance_threshold = 1 - correlation_threshold

        # Perform hierarchical clustering
        linked = linkage(condensed_distance_matrix, method=linkage_method)

        # Form clusters based on the distance threshold
        cluster_assignments = fcluster(
            linked, t=distance_threshold, criterion="distance"
        )

        if plot:
            plt.figure(figsize=(10, 7))
            dendrogram(linked, labels=corr_matrix.index.tolist())
            plt.axhline(
                y=distance_threshold, color="r", linestyle="--"
            )  # Visual cutoff line
            plt.title("Hierarchical Clustering Dendrogram")
            plt.xlabel("Stock")
            plt.ylabel("Distance (1 - Correlation)")
            plt.show()

        # Debug: Output clusters
        clusters = {}
        for stock, cluster_label in zip(returns_df.columns, cluster_assignments):
            clusters.setdefault(cluster_label, []).append(stock)

        # Identify correlated groups (clusters with more than one ticker)
        correlated_groups = [
            set(group) for group in clusters.values() if len(group) > 1
        ]

        if not correlated_groups:
            # print("No correlated groups found. Stopping iteration.")
            break

        # Select tickers to exclude based on Sharpe Ratio
        excluded_tickers = select_best_tickers(
            performance_df=performance_df,
            correlated_groups=correlated_groups,
            sharpe_threshold=sharpe_threshold,
        )

        # Debug: Check type and contents of excluded_tickers
        print(f"Excluded tickers (type {type(excluded_tickers)}): {excluded_tickers}")

        # If no tickers are excluded this iteration, break to avoid infinite loop
        if not excluded_tickers:
            # print("No more tickers to exclude. Stopping iteration.")
            break

        total_excluded.update(excluded_tickers)

        # Log tickers excluded in this iteration
        print(f"Iteration {iteration}: Excluded tickers: {excluded_tickers}")

        # Drop excluded tickers from the returns DataFrame
        returns_df = returns_df.drop(columns=excluded_tickers)

        iteration += 1

    # Log total excluded tickers
    print(f"Total excluded tickers: {total_excluded}")
    return returns_df.columns.tolist()


def select_best_tickers(performance_df, correlated_groups, sharpe_threshold=0.005):
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
        # logger.info(f"Maximum Sharpe Ratio in group: {max_sharpe:.4f}")

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
        # logger.info(f"Selected top {dynamic_n} tickers: {top_n} from group {group}")

        # Exclude other tickers in the group
        to_keep = set(top_n)
        to_exclude = group - to_keep
        tickers_to_exclude.update(to_exclude)

    return tickers_to_exclude
