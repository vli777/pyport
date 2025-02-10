from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from utils.performance_metrics import kappa_ratio, sharpe_ratio


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
    strategy_returns = (positions_df * returns_df).sum(axis=1).fillna(0)
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
