from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
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
        float: Kappa ratio. Returns np.inf if there is no downside risk.
    """
    excess_returns = returns - mar
    mean_excess = excess_returns.mean()
    # Compute lower partial moment (only include negative deviations)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return np.inf
    lpm = np.mean(np.abs(negative_returns) ** order)
    # Avoid division by zero
    if lpm == 0:
        return np.inf
    return mean_excess / (lpm ** (1 / order))


def simulate_strategy(
    returns_df: pd.DataFrame, positions_df: pd.DataFrame
) -> Tuple[pd.Series, dict]:
    """
    Simulates the strategy using positions and calculates performance metrics.
    Positions are assumed to be shifted (to avoid lookahead bias).

    Args:
        returns_df (pd.DataFrame): Daily log returns DataFrame.
        positions_df (pd.DataFrame): Positions DataFrame (tickers as columns, dates as index).

    Returns:
        tuple: (strategy_returns, metrics)
            strategy_returns (pd.Series): Daily strategy returns.
            metrics (dict): Dictionary with cumulative_return, sharpe, and kappa.
    """
    # Shift positions to avoid lookahead bias
    strategy_returns = (positions_df.shift(1) * returns_df).sum(axis=1).fillna(0)
    cumulative_return = (strategy_returns + 1).prod() - 1
    sr = sharpe_ratio(strategy_returns)
    kp = kappa_ratio(strategy_returns, order=3)
    metrics = {"cumulative_return": cumulative_return, "sharpe": sr, "kappa": kp}
    return strategy_returns, metrics


def composite_score(metrics: dict, weights: dict = None) -> float:
    """
    Combines performance metrics into a composite score.

    By default, the composite score is a weighted sum:
      40% cumulative_return + 30% sharpe_ratio + 30% kappa_ratio.

    Args:
        metrics (dict): Dictionary with keys "cumulative_return", "sharpe", "kappa".
        weights (dict, optional): Weights for each metric. Defaults to {"cumulative_return": 0.4, "sharpe": 0.3, "kappa": 0.3}.

    Returns:
        float: Composite performance score.
    """
    if weights is None:
        weights = {"cumulative_return": 0.4, "sharpe": 0.3, "kappa": 0.3}
    score = (
        weights["cumulative_return"] * metrics["cumulative_return"]
        + weights["sharpe"] * metrics["sharpe"]
        + weights["kappa"] * metrics["kappa"]
    )
    return score


def calculate_performance_metrics(
    returns_df: pd.DataFrame, risk_free_rate: float = 0.0
):
    """
    Compute Sharpe Ratios and Total Returns for each ticker.
    """
    daily_rf = risk_free_rate / 252
    means = returns_df.mean()
    stds = returns_df.std()
    excess = means - daily_rf

    # Debug log for a few tickers
    sample_tickers = returns_df.columns[:5]
    for ticker in sample_tickers:
        logger.debug(
            f"{ticker} - Mean: {means[ticker]:.6f}, Std: {stds[ticker]:.6f}, Excess Return: {excess[ticker]:.6f}"
        )

    # Handle cases where std is 0 (avoid division errors)
    sharpe_ratios = np.where(stds > 0, (excess / stds) * np.sqrt(252), np.nan)
    total_returns = (1 + returns_df).prod() - 1

    performance_df = pd.DataFrame(
        {"Sharpe Ratio": sharpe_ratios, "Total Return": total_returns},
        index=returns_df.columns,
    )

    return performance_df


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
