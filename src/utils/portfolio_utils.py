# src/utils/portfolio_utils.py

from typing import Any, Dict, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


from .logger import logger


def convert_to_dict(weights: Any, asset_names: list) -> Dict[str, float]:
    """
    Convert optimizer weights to a dictionary.
    Handles numpy.ndarray, pandas.DataFrame, and dict types.

    Args:
        weights (Any): The weights to convert. Can be ndarray, DataFrame, or dict.
        asset_names (List[str]): List of asset names corresponding to the weights.

    Returns:
        Dict[str, float]: A dictionary mapping asset names to their corresponding weights.

    Raises:
        ValueError: If the length of weights does not match the number of asset names.
        TypeError: If weights is not of type ndarray, DataFrame, or dict.
    """
    if isinstance(weights, np.ndarray):
        logger.debug("Converting weights from numpy.ndarray to dict.")
        if weights.ndim > 1:
            weights = weights.flatten()
            logger.debug(f"Flattened weights: {weights}")

        if len(weights) != len(asset_names):
            logger.error(
                "The number of weights does not match the number of asset names."
            )
            raise ValueError(
                "The number of weights does not match the number of asset names."
            )

        converted = {asset: weight for asset, weight in zip(asset_names, weights)}
        logger.debug(f"Converted weights: {converted}")
        return converted

    elif isinstance(weights, pd.DataFrame):
        logger.debug("Converting weights from pandas.DataFrame to dict.")
        if weights.shape[1] != 1:
            logger.error("DataFrame must have exactly one column for weights.")
            raise ValueError("DataFrame must have exactly one column for weights.")

        weights_series = weights.squeeze()
        if not weights_series.index.equals(pd.Index(asset_names)):
            logger.error(
                "Asset names in DataFrame index do not match the provided asset_names list."
            )
            raise ValueError(
                "Asset names in DataFrame index do not match the provided asset_names list."
            )

        converted = weights_series.to_dict()
        logger.debug(f"Converted weights: {converted}")
        return converted

    elif isinstance(weights, dict):
        logger.debug("Weights are already in dictionary format.")
        if set(weights.keys()) != set(asset_names):
            logger.warning(
                "The asset names in weights do not match the provided asset_names list."
            )
        return weights

    else:
        logger.error(
            f"Unsupported weight type: {type(weights)}. Must be ndarray, DataFrame, or dict."
        )
        raise TypeError("Unsupported weight type: must be ndarray, DataFrame, or dict.")


def normalize_weights(weights, min_weight: float) -> pd.Series:
    """
    Normalize the weights by filtering out values below min_weight and scaling the remaining weights to sum to 1.
    Additionally, rounds each weight to three decimal places.

    Args:
        weights (dict or pd.Series): The input weights.
        min_weight (float): The minimum weight threshold.

    Returns:
        pd.Series: The normalized weights with indices as asset symbols.
    """
    logger.debug(f"Original weights: {weights}")
    logger.debug(f"Minimum weight threshold: {min_weight}")

    # Convert dict to Series if necessary
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    # Filter out assets whose absolute weight is below min_weight
    import numbers

    filtered_weights = {
        k: float(v)
        for k, v in weights.items()
        if isinstance(v, numbers.Number) and abs(v) >= min_weight
    }
    logger.debug(f"Filtered weights: {filtered_weights}")

    total_weight = sum(filtered_weights.values())
    logger.debug(f"Total weight after filtering: {total_weight}")

    if total_weight == 0:
        logger.error("No weights remain after filtering with the specified min_weight.")
        raise ValueError(
            "No weights remain after filtering with the specified min_weight."
        )

    # Normalize the weights so they sum to 1
    normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}
    logger.debug(f"Normalized weights before rounding: {normalized_weights}")

    # Round each weight to three decimal places
    rounded_weights = {k: round(v, 3) for k, v in normalized_weights.items()}
    logger.debug(f"Normalized weights after rounding: {rounded_weights}")

    # Return a Pandas Series instead of a dict
    return pd.Series(rounded_weights)


def stacked_output(stack_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Flatten a dictionary of dictionaries and compute the arithmetic average of the values.
    Missing keys in any sub-dictionary are treated as having a value of 0.

    Args:
        stack_dict (Dict[str, Dict[str, float]]): A dictionary where each key maps to another dictionary of asset weights.

    Returns:
        Dict[str, float]: A single dictionary with averaged weights.

    Raises:
        ValueError: If stack_dict is empty or contains no valid portfolios.
    """
    logger.debug(f"Input stack_dict: {stack_dict}")

    if not stack_dict:
        logger.error("Input stack_dict is empty.")
        raise ValueError("Input stack_dict is empty.")

    # Collect all unique asset names across all portfolios
    all_assets = set()
    for portfolio in stack_dict.values():
        if isinstance(portfolio, dict):
            all_assets.update(portfolio.keys())
        else:
            logger.warning(f"Invalid portfolio format: {portfolio}. Expected a dict.")

    if not all_assets:
        logger.warning("No valid assets found in stack_dict.")
        # Instead of raising an exception, return an empty dict
        return {}

    logger.debug(f"All unique assets: {all_assets}")

    # Initialize a dictionary to accumulate weights
    total_weights = defaultdict(float)
    num_portfolios = len(stack_dict)
    logger.debug(f"Number of portfolios: {num_portfolios}")

    # Sum the weights for each asset, treating missing assets as 0
    for portfolio in stack_dict.values():
        if isinstance(portfolio, dict):
            for asset in all_assets:
                total_weights[asset] += portfolio.get(asset, 0.0)

    # Calculate the average weights
    average_weights = {
        asset: round(total_weight / num_portfolios, 3)
        for asset, total_weight in total_weights.items()
    }
    logger.debug(f"Average weights: {average_weights}")

    return average_weights


def holdings_match(
    cached_model_dict: Dict[str, Any], input_file_symbols: list, test_mode: bool = False
) -> bool:
    """
    Check if all selected symbols match between the cached model dictionary and the input file symbols.

    Args:
        cached_model_dict (Dict[str, Any]): Dictionary containing cached model symbols.
        input_file_symbols (List[str]): List of symbols from the input file.
        test_mode (bool, optional): If True, prints mismatched symbols. Defaults to False.

    Returns:
        bool: True if all symbols match, False otherwise.
    """
    cached_symbols = set(cached_model_dict.keys())
    input_symbols = set(input_file_symbols)

    missing_in_input = cached_symbols - input_symbols
    missing_in_cache = input_symbols - cached_symbols

    if missing_in_input:
        if test_mode:
            logger.warning(
                f"Symbols {missing_in_input} are in cached_model_dict but not in input_file_symbols."
            )
        else:
            logger.warning(
                f"Symbols {missing_in_input} are missing in input_file_symbols."
            )
        return False

    if missing_in_cache:
        if test_mode:
            logger.warning(
                f"Symbols {missing_in_cache} are in input_file_symbols but not in cached_model_dict."
            )
        else:
            logger.warning(
                f"Symbols {missing_in_cache} are missing in cached_model_dict."
            )
        return False

    logger.debug("All symbols match between cached_model_dict and input_file_symbols.")
    return True


def trim_weights(weights: Dict[str, float], max_size: int) -> Dict[str, float]:
    """
    Returns a new dict with only the largest `max_size` weights.
    No renormalization is done, assuming you already normalized.
    """
    if len(weights) <= max_size:
        return weights

    # Sort ascending by weight
    sorted_weights = sorted(weights.items(), key=lambda kv: kv[1])
    # Keep the largest `max_size` items
    trimmed = dict(sorted_weights[-max_size:])
    return trimmed


def calculate_portfolio_performance(
    data: pd.DataFrame, weights: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Given price data (rows = dates, columns = tickers) and a dict of weights,
    compute:
      1) daily log returns of each ticker
      2) weighted sum (i.e. portfolio daily returns)
      3) portfolio cumulative returns
      4) combined daily/cumulative returns for each ticker and the portfolio

    Returns:
      (returns, portfolio_returns, portfolio_cumulative_returns, combined_df)
    """
    # 1) Compute log returns
    returns = np.log(data) - np.log(data.shift(1))
    # Drop the first NaN row
    returns = returns.iloc[1:, :]

    # 2) Weighted returns and sum
    weights_vector = list(weights.values())
    weighted_returns = returns.mul(weights_vector, axis="columns")
    portfolio_returns = weighted_returns.sum(axis=1)

    # 3) Portfolio cumulative returns
    portfolio_cumulative_returns = (portfolio_returns + 1).cumprod()

    # 4) Combine daily & cumulative returns for optional plotting
    #    - daily returns of each ticker -> returns
    #    - daily returns of portfolio -> portfolio_returns
    #    - cumulative returns of each ticker -> returns.add(1).cumprod()
    #    - cumulative returns of portfolio -> portfolio_cumulative_returns

    portfolio_returns_df = portfolio_returns.to_frame(name="SIM_PORT")
    portfolio_cumulative_df = portfolio_cumulative_returns.to_frame(name="SIM_PORT")

    all_daily_returns = returns.join(portfolio_returns_df)
    all_cumulative_returns = (portfolio_cumulative_df - 1).join(
        returns.add(1).cumprod() - 1
    )

    return (
        returns,
        portfolio_returns,
        portfolio_cumulative_returns,
        (all_daily_returns, all_cumulative_returns),
    )


def sharpe_ratio(
    returns: pd.Series, entries_per_year: int = 252, risk_free_rate: float = 0.0
) -> float:
    """
    Calculates annualized Sharpe ratio for pd.Series of normal or log returns.

    Risk-free rate should be given for the same period the returns are given.
    For example, if the input returns are observed in 3 months, the risk-free
    rate given should be the 3-month risk-free rate.

    :param returns: (pd.Series) Returns - normal or log
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :param risk_free_rate: (float) Risk-free rate (0 by default)
    :return: (float) Annualized Sharpe ratio
    """
    excess_return = returns.mean() - risk_free_rate
    annualized_volatility = returns.std() * np.sqrt(entries_per_year)
    sharpe_r = excess_return / annualized_volatility

    return sharpe_r


def calculate_performance_metrics(returns_df, risk_free_rate=0.0):
    daily_rf = risk_free_rate / 252
    means = returns_df.mean()
    stds = returns_df.std()
    excess = means - daily_rf

    sample_tickers = returns_df.columns[:5]
    for ticker in sample_tickers:
        logger.debug(
            f"{ticker} - Mean: {means[ticker]:.6f}, Std: {stds[ticker]:.6f}, Excess Return: {excess[ticker]:.6f}"
        )

    sharpe_ratios = (excess / stds) * np.sqrt(252)
    total_returns = (1 + returns_df).prod() - 1

    for ticker in sample_tickers:
        logger.debug(
            f"{ticker} - Sharpe Ratio: {sharpe_ratios[ticker]:.4f}, Total Return: {total_returns[ticker]:.4f}"
        )

    performance_df = pd.DataFrame(
        {"Sharpe Ratio": sharpe_ratios, "Total Return": total_returns}
    )

    return performance_df


import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


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
        print(f"Iteration {iteration}: Diagonal of correlation matrix: {diag}")

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
        print(f"Iteration {iteration}: Diagonal of distance matrix: {diag_dist}")

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


def apply_kalman_filter(returns_series, threshold=7.0, epsilon=1e-6):
    # Ensure returns_series is 1-dimensional
    if not isinstance(returns_series, pd.Series):
        raise ValueError("returns_series must be a Pandas Series.")

    # Initialize the Kalman filter
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

    # Increase process noise for smoother estimates
    kf.transition_covariance = np.eye(1) * 0.1

    # Reshape the input to 2D array (T x 1)
    values = returns_series.values.reshape(-1, 1)

    # Train the Kalman filter
    kf = kf.em(values, n_iter=3)

    # Get smoothed state means
    smoothed_state_means, _ = kf.smooth(values)

    # Calculate residuals
    residuals = values - smoothed_state_means

    # Calculate the median of residuals
    median_res = np.median(residuals)

    # Compute the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(residuals - median_res))

    # Avoid division by zero or very small MAD
    mad = max(mad, epsilon)

    # Define a modified Z-score using MAD
    modified_z_scores = 0.6745 * (residuals - median_res) / mad

    # Identify anomalies based on a threshold on the modified Z-score
    anomaly_flags = np.abs(modified_z_scores) > threshold

    # Squeeze anomaly_flags to convert from 2D to 1D
    anomaly_flags = anomaly_flags.squeeze()

    return anomaly_flags


def plot_anomalies(stocks, returns_data, anomaly_flags_data, stocks_per_page=36):
    """
    Plots multiple stocks' return series in paginated 6x6 grids and highlights anomalies using Seaborn.

    Args:
        stocks (list): List of stock names.
        returns_data (dict): Dictionary of daily returns for each stock, keyed by stock name.
        anomaly_flags_data (dict): Dictionary of anomaly flags (np.array) for each stock, keyed by stock name.
        stocks_per_page (int): Maximum number of stocks to display per page (default is 36).
    """
    grid_rows, grid_cols = 6, 6  # Fixed grid dimensions
    total_plots = grid_rows * grid_cols
    num_pages = (
        len(stocks) + stocks_per_page - 1
    ) // stocks_per_page  # Calculate pages

    for page in range(num_pages):
        # Determine the range of stocks for this page
        start_idx = page * stocks_per_page
        end_idx = min(start_idx + stocks_per_page, len(stocks))
        stocks_to_plot = stocks[start_idx:end_idx]

        # Create a figure for this page
        fig, axes = plt.subplots(
            grid_rows, grid_cols, figsize=(18, 18), sharex=False, sharey=False
        )
        axes = axes.flatten()  # Flatten for easier indexing

        for i, stock in enumerate(stocks_to_plot):
            ax = axes[i]

            # Extract data for this stock using dictionary keys
            if stock in returns_data:
                returns_series = returns_data[stock]
            else:
                print(f"Warning: {stock} not found in returns_data.")
                continue  # Skip this stock if data is missing

            anomaly_flags = anomaly_flags_data[stock]

            # Convert anomaly_flags to a Pandas Series aligned with returns_series
            anomaly_flags_series = pd.Series(anomaly_flags, index=returns_series.index)

            # Initialize Kalman filter and perform smoothing
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            values = returns_series.values.reshape(-1, 1)
            kf = kf.em(values, n_iter=10)
            smoothed_state_means, smoothed_state_covariances = kf.smooth(values)

            # Calculate 95% confidence intervals
            mean = smoothed_state_means.squeeze()
            std_dev = np.sqrt(smoothed_state_covariances.squeeze())
            lower_bounds = mean - 1.96 * std_dev
            upper_bounds = mean + 1.96 * std_dev

            # Create a DataFrame for easier plotting
            plot_df = pd.DataFrame(
                {
                    "Date": returns_series.index,
                    "Returns": returns_series.values,
                    "Anomaly": anomaly_flags_series,
                }
            )

            # Plot the observed returns
            sns.lineplot(
                ax=ax,
                data=plot_df,
                x="Date",
                y="Returns",
                color="blue",
                label="Observed Returns",
            )

            # Overlay the Kalman smoothed mean
            ax.plot(plot_df["Date"], mean, color="green", label="Kalman Smoothed Mean")

            # Fill between the confidence intervals
            ax.fill_between(
                plot_df["Date"],
                lower_bounds,
                upper_bounds,
                color="gray",
                alpha=0.3,
                label="95% Confidence Interval",
            )

            # Highlight anomalies
            sns.scatterplot(
                ax=ax,
                data=plot_df[plot_df["Anomaly"]],
                x="Date",
                y="Returns",
                color="red",
                s=20,
                label="Anomalies",
            )

            # Customize each subplot
            ax.set_title(stock, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend(fontsize=8)
            ax.grid(True)

        # Remove unused subplots
        for j in range(len(stocks_to_plot), total_plots):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.suptitle(f"Page {page + 1} of {num_pages}", fontsize=16)
        plt.show()


def remove_anomalous_stocks(returns_df, threshold=7.0, plot=False):
    """
    Removes stocks with anomalous returns based on the Kalman filter.
    Optionally plots anomalies for all flagged stocks in a paginated 6x6 grid.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns.
        threshold (float): Number of standard deviations to flag anomalies.
        plot (bool): If True, anomalies will be plotted in a paginated grid.

    Returns:
        pd.DataFrame: Filtered DataFrame with anomalous stocks removed.
    """
    anomalous_cols = []

    # Dictionaries to store data for plotting if needed
    returns_data = {}
    anomaly_flags_data = {}

    for stock in returns_df.columns:
        returns_series = returns_df[stock]
        anomaly_flags = apply_kalman_filter(returns_series, threshold=threshold)

        # If anomalies found for the stock
        if anomaly_flags.any():
            anomalous_cols.append(stock)
            # Store data for plotting if plot is True
            if plot:
                returns_data[stock] = returns_series
                anomaly_flags_data[stock] = anomaly_flags

    print(
        f"Removing {len(anomalous_cols)} stocks with Kalman anomalies: {anomalous_cols}"
    )

    # If plotting is requested and there are stocks with anomalies, plot them
    if plot and returns_data:
        # Use the list of anomalous stocks for plotting
        plot_anomalies(
            stocks=anomalous_cols,
            returns_data=returns_data,
            anomaly_flags_data=anomaly_flags_data,
            stocks_per_page=36,
        )

    # Return the DataFrame with anomalous stocks removed
    return returns_df.drop(columns=anomalous_cols)


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


def limit_portfolio_size(
    weights: pd.Series, max_holdings: int, target_sum: float
) -> pd.Series:
    """
    Limit the portfolio to top N holdings by absolute weight and normalize the weights to a target sum.

    Args:
        weights (pd.Series): Series of asset weights (can include negatives for short positions).
        max_holdings (int): Maximum number of assets to retain.
        target_sum (float): The desired total sum for the normalized weights.

    Returns:
        pd.Series: The portfolio weights limited to top holdings and normalized to the target sum.
    """
    # Select top N holdings by absolute weight
    top_holdings = weights.abs().nlargest(max_holdings).index
    limited_weights = weights.loc[top_holdings]

    # Normalize to the specified target sum
    current_sum = limited_weights.sum()
    if current_sum != 0:
        limited_weights = limited_weights / current_sum * target_sum

    return limited_weights
