import sys
from typing import Any, Dict, Optional
import pandas as pd

from config import Config
from utils.performance_metrics import calculate_portfolio_performance, sharpe_ratio
from utils import logger
from utils.portfolio_utils import (
    trim_weights,
)


def output(
    data: pd.DataFrame,
    allocation_weights: Dict[str, float],
    start_date: Any,
    end_date: Any,
    inputs: Optional[str] = None,
    optimization_model: Optional[str] = None,
    time_period: float = 1.0,
    config: Config = None,
):
    """
    Produces console output (stats) and returns daily/cumulative returns of
    the portfolio + each ticker.
    """

    # Ensure allocation_weights is a dictionary
    clean_weights = (
        allocation_weights
        if isinstance(allocation_weights, dict)
        else dict(allocation_weights)
    )

    # Validate that data contains all required symbols
    missing_symbols = [
        symbol for symbol in clean_weights.keys() if symbol not in data.columns
    ]
    if missing_symbols:
        logger.warning(
            f"The following symbols are missing in the data and will be filled with zeros: {missing_symbols}"
        )
        # Add missing columns with zero values
        for symbol in missing_symbols:
            data[symbol] = 0

    # Trim if we have more assets than allowed
    if len(clean_weights) > config.portfolio_max_size:
        clean_weights = trim_weights(clean_weights, config.portfolio_max_size)

    # If nothing left, can't proceed
    if len(clean_weights) == 0:
        logger.info("No assets left after trimming or filtering. Exiting.")
        sys.exit()

    # --- Calculate portfolio performance
    (
        returns,
        portfolio_returns,
        portfolio_cumulative_returns,
        (all_daily_returns, all_cumulative_returns),
    ) = calculate_portfolio_performance(data[list(clean_weights.keys())], clean_weights)

    try:
        sharpe = sharpe_ratio(portfolio_returns, risk_free_rate=config.risk_free_rate)
    except ZeroDivisionError:
        sharpe = 0

    # logger.info stats
    if inputs is not None:
        logger.info(f"\n\nWatchlist Inputs: {inputs}")
    logger.info(f"\nTime period: {start_date} to {end_date} ({time_period} yrs)")

    cumulative_pct = round((portfolio_cumulative_returns.iloc[-1] - 1) * 100, 2)
    logger.info(f"Optimization method: {optimization_model}")
    logger.info(f"Sharpe ratio: {round(sharpe, 2)}")
    logger.info(f"Cumulative return: {cumulative_pct}%")
    logger.info(f"Portfolio allocation weights (min {config.min_weight:.2f}):")

    sorted_weights = sorted(clean_weights.items(), key=lambda kv: kv[1], reverse=True)
    for symbol, weight in sorted_weights:
        logger.info(f"{symbol} \t{weight:.3f}")

    return all_daily_returns, all_cumulative_returns


def output_results(
    df: pd.DataFrame,
    weights: Dict[str, float],
    model_name: str,
    start_date: Any,
    end_date: Any,
    years: float,
    config: Config,
) -> None:
    """
    Wrapper around `output` for portfolio optimization results.
    Handles logging and reporting without returning performance data.
    """
    output(
        data=df,
        allocation_weights=weights,
        inputs=model_name,
        start_date=start_date,
        end_date=end_date,
        optimization_model=model_name,
        time_period=years,
        config=config,
    )
