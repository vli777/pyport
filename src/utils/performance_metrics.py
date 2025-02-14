from typing import Any, Dict, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import trim_mean

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

    if annualized_volatility == 0:
        return np.nan
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


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    min_obs: int = 5,
    trim_fraction: float = 0.1,
) -> float:
    """
    Calculates the Omega ratio of a return series using robust statistics.

    Omega = (Trimmed mean of gains above threshold) / (Trimmed mean of losses below threshold)

    Args:
        returns (pd.Series): Daily strategy returns.
        threshold (float): Minimum acceptable return (default 0 for break-even).
        min_obs (int): Minimum number of observations required for gains/losses.
        trim_fraction (float): Fraction of extreme values to trim from mean calculation.

    Returns:
        float: Omega ratio. Returns np.nan if insufficient data.
    """
    if returns.empty:
        return np.nan  # No returns available

    # Boolean masks for gains and losses
    gains_mask = returns > threshold
    losses_mask = returns < threshold

    num_gains = np.sum(gains_mask)
    num_losses = np.sum(losses_mask)

    # If not enough data points, return NaN
    if num_gains < min_obs or num_losses < min_obs:
        return np.nan

    # Compute robust gain: trimmed mean or median fallback
    gains = returns[gains_mask]
    robust_gain = (
        trim_mean(gains, proportiontocut=trim_fraction)
        if len(gains) > 2
        else np.median(gains)
    )

    # Compute robust loss: trimmed mean of absolute losses
    losses = returns[losses_mask]
    robust_loss = (
        -trim_mean(losses, proportiontocut=trim_fraction)
        if len(losses) > 2
        else -np.median(losses)
    )

    # Avoid division by near-zero losses
    if robust_loss < 1e-8:
        robust_loss = 1e-8

    return robust_gain / robust_loss


def conditional_var(
    returns: Union[np.ndarray, pd.Series], alpha: float = 0.05
) -> float:
    """
    Calculates Conditional Value at Risk (CVaR), also known as Expected Shortfall,
    for a given return series. If there is insufficient data, it returns 0.0
    (i.e. no additional penalty).

    Args:
        returns (Union[np.ndarray, pd.Series]): Daily strategy returns.
        alpha (float): Tail probability (default 5% worst losses).

    Returns:
        float: The CVaR value (a negative number for losses). If insufficient data,
               returns 0.0.
    """
    # Convert to pandas Series if needed
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # If not enough data, return 0.0 (no penalty)
    if returns.empty or len(returns) < 2:
        return 0.0

    # Sort returns in ascending order (worst returns first)
    sorted_losses = returns.sort_values(ascending=True)
    threshold_index = int(np.ceil(alpha * len(sorted_losses)))
    if threshold_index < 1:
        return 0.0

    tail_losses = sorted_losses.iloc[:threshold_index]
    return tail_losses.mean()


def calculate_portfolio_alpha(
    portfolio_returns: pd.DataFrame | pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate the portfolio's alpha using the CAPM model.

    Args:
        portfolio_returns (pd.DataFrame or pd.Series): Portfolio returns.
        market_returns (pd.Series): Market index returns.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.

    Returns:
        float: Portfolio alpha.
    """
    if portfolio_returns.empty or market_returns.empty:
        print("Warning: Filtered or market returns are empty. Returning alpha=0.0")
        return 0.0

    # Ensure portfolio_returns is a Series
    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns.mean(axis=1)  # Take mean across assets
    elif not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a DataFrame or Series.")

    # Align market_returns with portfolio_returns and forward-fill missing data
    market_returns = market_returns.reindex(portfolio_returns.index).ffill()

    # Drop any rows where either series has NaN values
    combined_data = pd.DataFrame(
        {"portfolio": portfolio_returns, "market": market_returns}
    ).dropna()

    if combined_data.empty:
        print(
            "Warning: After alignment, portfolio and market returns have no valid data. Returning alpha=0.0"
        )
        return 0.0

    # Extract cleaned excess returns
    excess_portfolio_returns = combined_data["portfolio"] - risk_free_rate
    excess_market_returns = combined_data["market"] - risk_free_rate

    # Fit CAPM model
    model = LinearRegression()
    model.fit(
        excess_market_returns.values.reshape(-1, 1), excess_portfolio_returns.values
    )
    alpha = model.intercept_

    return alpha


def portfolio_volatility(
    portfolio_returns: pd.Series, annualization_factor=252
) -> float:
    """
    Compute annualized portfolio volatility.
    """
    return portfolio_returns.std() * np.sqrt(annualization_factor)


def max_drawdown(portfolio_cumulative_returns: pd.Series) -> float:
    """
    Compute Maximum Drawdown (Max peak-to-trough decline).
    """
    peak = portfolio_cumulative_returns.cummax()
    drawdown = (portfolio_cumulative_returns - peak) / peak
    return drawdown.min()  # Max negative drawdown


def time_under_water(portfolio_cumulative_returns: pd.Series) -> int:
    """
    Compute Time Under Water (How long portfolio stays below peak).
    """
    peak = portfolio_cumulative_returns.cummax()
    underwater = portfolio_cumulative_returns < peak
    return underwater.sum()  # Total number of days underwater


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
    portfolio_portfolio_returns = portfolio_returns.to_frame(name="SIM_PORT")
    portfolio_cumulative_df = portfolio_cumulative_returns.to_frame(name="SIM_PORT")

    all_daily_returns = returns.join(portfolio_portfolio_returns)
    all_cumulative_returns = (portfolio_cumulative_df - 1).join(
        returns.add(1).cumprod() - 1
    )

    return (
        returns,
        portfolio_returns,
        portfolio_cumulative_returns,
        (all_daily_returns, all_cumulative_returns),
    )


def risk_return_contributions(
    weights: np.ndarray, daily_returns: pd.DataFrame, cumulative_returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the return and risk contributions for each stock in the portfolio.

    Parameters:
        weights (np.ndarray): Array of portfolio weights for each stock.
        daily_returns (pd.DataFrame): DataFrame of daily returns for each stock.
        cumulative_returns (np.ndarray): Array of cumulative returns for each stock.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized return and risk contributions as percentages.
    """
    # Compute standard deviation for risk
    stock_risks = daily_returns.std(axis=0).values  # Std dev of each stock

    # Compute return contributions
    return_contributions = weights * cumulative_returns
    return_contributions_pct = (
        return_contributions / np.sum(return_contributions) * 100
    )  # Normalize

    # Compute risk contributions
    risk_contributions = weights * stock_risks
    risk_contributions_pct = (
        risk_contributions / np.sum(risk_contributions) * 100
    )  # Normalize

    return return_contributions_pct, risk_contributions_pct
