from pathlib import Path
import sys
from typing import Any, Dict, Optional
import pandas as pd

from config import Config
from utils.performance_metrics import (
    calculate_portfolio_alpha,
    calculate_portfolio_performance,
    kappa_ratio,
    max_drawdown,
    portfolio_volatility,
    sharpe_ratio,
    time_under_water,
)
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

    sharpe = sharpe_ratio(portfolio_returns, risk_free_rate=config.risk_free_rate)
    kappa = kappa_ratio(portfolio_returns)
    volatility = portfolio_volatility(portfolio_returns)
    max_dd = max_drawdown(portfolio_cumulative_returns)
    time_uw = time_under_water(portfolio_cumulative_returns)

    # --- Load market data and compute market returns over the same period
    market_file = Path(config.data_dir) / "SPY.parquet"
    try:
        market_data = pd.read_parquet(market_file)
    except Exception as e:
        logger.error(f"Error loading market data from {market_file}: {e}")
        raise

    if "Adj Close" in market_data.columns:
        market_returns = market_data["Adj Close"].pct_change().dropna()
    elif "Close" in market_data.columns:
        market_returns = market_data["Close"].pct_change().dropna()
    else:
        raise ValueError("{market_file} data missing 'Adj Close' or 'Close' columns.")

    # Filter market_returns to match the analysis period
    # market_returns = market_returns.loc[start_date:end_date]
    # alpha = calculate_portfolio_alpha(
    #     portfolio_returns=portfolio_returns, market_returns=market_returns
    # )

    cumulative_pct = round((portfolio_cumulative_returns.iloc[-1] - 1) * 100, 2)

    # Logging results (using print for cleaner output)
    if inputs is not None:
        print("\n\nWatchlist Inputs:\n", inputs)

    print("=" * 50)
    print(f"Time Period:\t{start_date} to {end_date} ({time_period} yrs)")
    print(f"Optimization Method:\t{optimization_model}")
    print("=" * 50)

    print(f"Sharpe Ratio:\t\t{sharpe:.2f}")
    print(f"Kappa Ratio:\t\t{kappa:.2f}")
    print(f"Portfolio Volatility:\t{volatility * 100:.2f}%")
    print(f"Max Drawdown:\t\t{max_dd * 100:.2f}%")
    print(f"Time Under Water:\t{time_uw} days")
    # print(f"Portfolio Alpha vs Market:\t{alpha:.4f}")

    print(f"Cumulative Return:\t{cumulative_pct:.2f}%")
    print("=" * 50)

    # Print portfolio allocation in tab-separated format
    print("Asset\tWeight")  # Column headers
    sorted_weights = sorted(clean_weights.items(), key=lambda kv: kv[1], reverse=True)
    for asset, weight in sorted_weights:
        print(f"{asset}\t{weight:.4f}")

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
