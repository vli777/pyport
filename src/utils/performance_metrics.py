from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from .logger import logger


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


def kappa_ratio(returns: pd.Series, order: int = 3, mar: float = 0.0) -> float:
    """
    Calculate the Kappa ratio of a return series.
    Kappa_n = (mean(returns - MAR)) / (LPM_n)^(1/n)
    where LPM_n is the lower partial moment of order n (only returns below MAR contribute).

    Args:
        returns (pd.Series): Daily strategy returns.
        order (int): The order of the Kappa ratio (default 3, i.e. Kappa-3).
        mar (float): Minimum acceptable return (default 0).

    Returns:
        float: Kappa ratio. Returns np.nan if there is no downside risk.
    """
    if returns.empty:
        return np.nan  # No returns available

    excess_returns = returns - mar
    mean_excess = excess_returns.mean()

    # Compute lower partial moment (only include negative deviations)
    negative_returns = excess_returns[excess_returns < 0]

    if negative_returns.empty:
        return np.nan  # No downside risk, return NaN to avoid bias

    lpm = np.mean(np.abs(negative_returns) ** order)

    # Avoid division by zero issues
    if lpm == 0:
        return np.nan

    return mean_excess / (lpm ** (1 / order))


def calculate_portfolio_alpha(
    filtered_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate the portfolio's alpha using the CAPM model.

    Args:
        filtered_returns (pd.DataFrame): Returns of filtered tickers.
        market_returns (pd.Series): Market index returns.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.

    Returns:
        float: Portfolio alpha.
    """
    if filtered_returns.empty or market_returns.empty:
        logger.warning("Filtered or market returns are empty. Returning alpha=0.0")
        return 0.0

    # Compute portfolio return dynamically
    portfolio_returns = filtered_returns.mean(axis=1)

    # Align market_returns with portfolio_returns and **forward-fill missing data**
    market_returns = market_returns.reindex(portfolio_returns.index).ffill()

    if portfolio_returns.empty or market_returns.empty:
        logger.warning(
            "After alignment, portfolio returns or market returns are empty. Returning alpha=0.0"
        )
        return 0.0

    # Excess returns
    excess_portfolio_returns = portfolio_returns - risk_free_rate
    excess_market_returns = market_returns - risk_free_rate

    # Fit CAPM model
    model = LinearRegression()
    model.fit(
        excess_market_returns.values.reshape(-1, 1), excess_portfolio_returns.values
    )
    alpha = model.intercept_

    logger.debug(f"Calculated alpha: {alpha}")
    return alpha


def calculate_portfolio_performance(
    data: pd.DataFrame, weights: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Compute:
      1) daily log returns of each ticker
      2) weighted sum (i.e. portfolio daily returns)
      3) portfolio cumulative returns
      4) combined daily/cumulative returns for each ticker and the portfolio

    Args:
      data (pd.DataFrame): Multiindex DataFrame (rows = dates, columns = tickers)
      weights (Dict[str, float]): Portfolio allocation per ticker

    Returns:
      (returns, portfolio_returns, portfolio_cumulative_returns, combined_df)
    """
    # 1) Compute log returns
    returns = np.log(data) - np.log(data.shift(1))
    returns = returns.iloc[1:, :]  # Drop first NaN row

    # Ensure weights are only applied to available stocks on each date
    aligned_weights = returns.notna().astype(float).mul(pd.Series(weights), axis=1)
    aligned_weights = aligned_weights.div(aligned_weights.sum(axis=1), axis=0).fillna(0)

    logger.debug(f"Returns shape: {returns.shape}")
    logger.debug(f"Weights vector shape: {aligned_weights.shape}")

    # 2) Compute weighted portfolio returns dynamically
    portfolio_returns = (returns * aligned_weights).sum(axis=1)

    # 3) Portfolio cumulative returns
    portfolio_cumulative_returns = (portfolio_returns + 1).cumprod()

    # 4) Combine daily & cumulative returns
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
