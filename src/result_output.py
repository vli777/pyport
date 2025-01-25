import sys
from typing import Any, Dict, Optional
import pandas as pd

from config import Config
from utils import logger
from utils.portfolio_utils import (
    calculate_portfolio_performance,
    sharpe_ratio,
    trim_weights,
)


def output(
    data: pd.DataFrame,
    allocation_weights: Dict[str, float],
    start_date: Any,
    end_date: Any,
    inputs: Optional[str] = None,
    max_size: int = 10,
    optimization_model: Optional[str] = None,
    time_period: float = 1.0,
    minimum_weight: float = 0.01,
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

    if config and getattr(config, "test_mode", False):
        logger.info("Raw weights:", clean_weights)

    # Trim if we have more assets than allowed
    if len(clean_weights) > max_size:
        clean_weights = trim_weights(clean_weights, max_size)

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
    ) = calculate_portfolio_performance(data[clean_weights.keys()], clean_weights)

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
    logger.info(f"Portfolio allocation weights (min {minimum_weight:.2f}):")

    sorted_weights = sorted(clean_weights.items(), key=lambda kv: kv[1], reverse=True)
    for symbol, weight in sorted_weights:
        logger.info(f"{symbol} \t{weight:.3f}")

    return all_daily_returns, all_cumulative_returns


def output_results(df, weights, model_name, config, start_date, end_date, years):
    output(
        data=df,
        allocation_weights=weights,
        inputs=", ".join([str(i) for i in sorted(config.input_files)]),
        start_date=start_date,
        end_date=end_date,
        optimization_model=model_name,
        time_period=years,
        minimum_weight=config.min_weight,
        max_size=getattr(config, "portfolio_max_size", 10),
        config=config,
    )
