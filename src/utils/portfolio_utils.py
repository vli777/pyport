# src/utils/portfolio_utils.py

from typing import Any, Dict, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import seaborn as sns
import matplotlib.pyplot as plt

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


def normalize_weights(weights: Dict[str, float], min_weight: float) -> Dict[str, float]:
    """
    Normalize the weights by:
      1. Filtering out assets where the absolute weight is below min_weight.
      2. Scaling the remaining weights so that their sum equals 1.
      3. Rounding each weight to three decimal places.

    This function now supports negative weights by using the absolute value for filtering.

    Args:
        weights (Dict[str, float]): The input weights dictionary, possibly containing negative values.
        min_weight (float): The minimum absolute weight threshold. Weights with |weight| below this value are filtered out.

    Returns:
        Dict[str, float]: The normalized weights dictionary.

    Raises:
        ValueError: If no weights remain after filtering or if normalization cannot be performed.
    """
    logger.debug(f"Original weights: {weights}")
    logger.debug(f"Minimum weight threshold: {min_weight}")

    # Filter out assets whose absolute weight is below min_weight, preserving sign for others
    filtered_weights = {k: v for k, v in weights.items() if abs(v) >= min_weight}
    logger.debug(f"Filtered weights: {filtered_weights}")

    # Sum the remaining weights (which may include negative values)
    total_weight = sum(filtered_weights.values())
    logger.debug(f"Total weight after filtering: {total_weight}")

    # Check if the sum of weights is effectively zero, which prevents normalization
    if total_weight == 0:
        logger.error(
            "Sum of weights is zero after filtering; cannot normalize weights."
        )
        raise ValueError(
            "No weights remain after filtering with the specified min_weight, or their sum is zero."
        )

    # Normalize the remaining weights so that they sum to 1, preserving the sign of each weight
    normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}
    logger.debug(f"Normalized weights before rounding: {normalized_weights}")

    # Round each weight to three decimal places
    rounded_weights = {k: round(v, 3) for k, v in normalized_weights.items()}
    logger.debug(f"Normalized weights after rounding: {rounded_weights}")

    return rounded_weights


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
            continue

    if not all_assets:
        logger.error("No valid assets found in stack_dict.")
        raise ValueError("No valid assets found in stack_dict.")

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


def identify_correlated_groups(corr_matrix, threshold=0.8):
    """
    Identifies groups of highly correlated tickers.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        threshold (float): Correlation threshold to define groups.

    Returns:
        list of sets: Groups of correlated tickers.
    """
    # Find pairs above the correlation threshold
    all_pairs = corr_matrix.stack()[lambda x: x > threshold].index.tolist()

    # Build groups of correlated tickers
    groups = []
    for ticker1, ticker2 in all_pairs:
        found = False
        for group in groups:
            if ticker1 in group or ticker2 in group:
                group.update([ticker1, ticker2])
                found = True
                break
        if not found:
            groups.append(set([ticker1, ticker2]))

    return groups


def filter_correlated_groups(
    returns_df, performance_df, sharpe_threshold=0.005, correlation_threshold=0.8, n=1
):
    """
    Iteratively filter correlated tickers based on correlation and Sharpe Ratio.
    """
    total_excluded = set()
    iteration = 1

    while True:
        # Compute the correlation matrix
        corr_matrix = returns_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)

        # Identify correlated groups
        groups = identify_correlated_groups(
            corr_matrix, threshold=correlation_threshold
        )

        if not groups:  # Exit if no groups found
            break

        # Use Sharpe-based selection to decide tickers to exclude
        excluded_tickers = select_best_tickers(
            performance_df=performance_df,
            correlated_groups=groups,
            sharpe_threshold=sharpe_threshold,
            n=n,
        )
        total_excluded.update(excluded_tickers)

        # Log tickers excluded in this iteration
        print(f"Iteration {iteration}: Excluded tickers: {excluded_tickers}")

        # Drop excluded tickers from the returns DataFrame
        returns_df = returns_df.drop(columns=excluded_tickers)

        iteration += 1

    # Log total excluded tickers
    print(f"Total excluded tickers: {total_excluded}")
    return returns_df.columns.tolist()


def apply_kalman_filter(returns_series, threshold=7.0):
    # Ensure returns_series is 1-dimensional
    if not isinstance(returns_series, pd.Series):
        raise ValueError("returns_series must be a Pandas Series.")

    # Initialize the Kalman filter
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

    # Reshape the input to 2D array (T x 1)
    values = returns_series.values.reshape(-1, 1)

    # Train the Kalman filter
    kf = kf.em(values, n_iter=10)

    # Get smoothed state means
    smoothed_state_means, _ = kf.smooth(values)

    # Calculate residuals
    residuals = values - smoothed_state_means

    # Calculate the median of residuals
    median_res = np.median(residuals)

    # Compute the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(residuals - median_res))

    # Define a modified Z-score using MAD (constant 0.6745 makes it comparable to standard deviation under normality)
    modified_z_scores = 0.6745 * (residuals - median_res) / mad

    # Identify anomalies based on a threshold on the modified Z-score
    anomaly_flags = np.abs(modified_z_scores) > threshold

    # Squeeze anomaly_flags to convert from 2D to 1D
    anomaly_flags = anomaly_flags.squeeze()

    return anomaly_flags


def plot_anomalies(stock, returns_series, anomaly_flags):
    """
    Plots the stock's return series and highlights anomalies using Seaborn.

    Args:
        stock (str): The name of the stock.
        returns_series (pd.Series): Daily returns for the stock.
        anomaly_flags (np.array): Boolean array indicating anomalies.
    """
    # Convert anomaly_flags to a Pandas Series aligned with returns_series
    anomaly_flags_series = pd.Series(anomaly_flags, index=returns_series.index)

    # Create a DataFrame for easier plotting
    plot_df = pd.DataFrame(
        {
            "Date": returns_series.index,
            "Returns": returns_series.values,
            "Anomaly": anomaly_flags_series,  # True/False for anomalies
        }
    )

    # Initialize Kalman filter and perform smoothing to get predicted range
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    values = returns_series.values.reshape(-1, 1)  # Reshape for KalmanFilter input
    kf = kf.em(values, n_iter=10)
    smoothed_state_means, smoothed_state_covariances = kf.smooth(values)

    # Calculate 95% confidence intervals from the smoothed estimates
    mean = smoothed_state_means.squeeze()
    std_dev = np.sqrt(smoothed_state_covariances.squeeze())
    lower_bounds = mean - 1.96 * std_dev
    upper_bounds = mean + 1.96 * std_dev

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Plot the observed returns
    sns.lineplot(
        data=plot_df, x="Date", y="Returns", label="Observed Returns", color="blue"
    )

    # Overlay the Kalman smoothed mean
    plt.plot(plot_df["Date"], mean, label="Kalman Smoothed Mean", color="green")

    # Fill between the confidence intervals
    plt.fill_between(
        plot_df["Date"],
        lower_bounds,
        upper_bounds,
        color="gray",
        alpha=0.3,
        label="95% Confidence Interval",
    )

    # Highlight anomalies
    sns.scatterplot(
        data=plot_df[plot_df["Anomaly"]],  # Filter rows with anomalies
        x="Date",
        y="Returns",
        color="red",
        label="Anomalies",
        s=50,  # Size of anomaly points
    )

    # Customize the plot
    plt.title(f"Anomalies and Kalman Range for {stock}")
    plt.xlabel("Date")
    plt.ylabel("Daily Returns")
    plt.legend()
    plt.grid(True)
    plt.show()


def remove_anomalous_stocks(returns_df, threshold=7.0, plot=False):
    """
    Removes stocks with anomalous returns based on the Kalman filter.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns.
        threshold (float): Number of standard deviations to flag anomalies.
        plot (bool): Bool if the anomalies should be plotted, default False

    Returns:
        pd.DataFrame: Filtered DataFrame with anomalous stocks removed.
    """
    anomalous_cols = []
    for stock in returns_df.columns:
        returns_series = returns_df[stock]
        anomaly_flags = apply_kalman_filter(returns_series, threshold=threshold)

        if anomaly_flags.any():
            anomalous_cols.append(stock)
            # Plot anomalies for visual inspection
            if plot:
                plot_anomalies(stock, returns_series, anomaly_flags)

    print(
        f"Removing {len(anomalous_cols)} stocks with Kalman anomalies: {anomalous_cols}"
    )
    return returns_df.drop(columns=anomalous_cols)


def select_best_tickers(performance_df, correlated_groups, sharpe_threshold=0.005, n=1):
    """
    Select top N tickers from each correlated group based on Sharpe Ratio and Total Return.

    Args:
        performance_df (pd.DataFrame): DataFrame with performance metrics.
        correlated_groups (list of sets): Groups of highly correlated tickers.
        sharpe_threshold (float): Threshold to consider Sharpe ratios as similar.
        n (int): Number of top tickers to select per group.

    Returns:
        set: Tickers to exclude.
    """
    tickers_to_keep = set()

    for group in correlated_groups:
        logger.info(f"Evaluating group of correlated tickers: {group}")
        group_metrics = performance_df.loc[list(group)]

        max_sharpe = group_metrics["Sharpe Ratio"].max()
        logger.info(f"Maximum Sharpe Ratio in group: {max_sharpe:.4f}")

        # Identify tickers within the Sharpe threshold of the max
        top_candidates = group_metrics[
            group_metrics["Sharpe Ratio"] >= (max_sharpe - sharpe_threshold)
        ]

        # Select top n based on Total Return among these
        top_n = top_candidates.nlargest(n, "Total Return").index.tolist()
        logger.info(f"Selected top {n} tickers: {top_n} from group {group}")

        tickers_to_keep.update(top_n)

    # Aggregate all tickers from all groups
    all_group_tickers = set()
    for group in correlated_groups:
        all_group_tickers.update(group)

    # Determine tickers to exclude: those in groups but not among top keepers
    tickers_to_exclude = all_group_tickers - tickers_to_keep
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
