# src/utils/portfolio_utils.py

from typing import Any, Dict, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
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
    Normalize the weights by filtering out values below min_weight and scaling the remaining weights to sum to 1.
    Additionally, rounds each weight to three decimal places.

    Args:
        weights (Dict[str, float]): The input weights dictionary.
        min_weight (float): The minimum weight threshold. Weights below this value are filtered out.

    Returns:
        Dict[str, float]: The normalized weights dictionary.

    Raises:
        ValueError: If no weights remain after filtering.
    """
    logger.debug(f"Original weights: {weights}")
    logger.debug(f"Minimum weight threshold: {min_weight}")

    # Filter out weights below min_weight
    filtered_weights = {k: v for k, v in weights.items() if v >= min_weight}
    logger.debug(f"Filtered weights: {filtered_weights}")

    total_weight = sum(filtered_weights.values())
    logger.debug(f"Total weight after filtering: {total_weight}")

    if total_weight == 0:
        logger.error("No weights remain after filtering with the specified min_weight.")
        raise ValueError(
            "No weights remain after filtering with the specified min_weight."
        )

    # Normalize the remaining weights to sum to 1
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
    returns: pd.Series, entries_per_year: int = 252, risk_free_rate: float = 0
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
    # Calculate the average return
    excess_return = returns.mean() - risk_free_rate

    # Adjust the standard deviation for annualization
    annualized_volatility = returns.std() * np.sqrt(entries_per_year)

    # Return the Sharpe ratio
    sharpe_r = excess_return / annualized_volatility

    return sharpe_r
