from typing import Tuple, Dict

import numpy as np
import pandas as pd

from utils.performance_metrics import (
    conditional_var,
    kappa_ratio,
    omega_ratio,
    sharpe_ratio,
)


def strategy_performance_metrics(
    returns_df: pd.DataFrame,
    positions_df: pd.DataFrame = None,  # Optional for portfolio simulation
    risk_free_rate: float = 0.0,
    objective_weights: Dict[str, float] = None,
) -> pd.Series:
    """
    Computes performance metrics for individual assets or a full strategy.

    If `positions_df` is provided, it simulates the portfolio returns.
    Otherwise, it computes per-asset metrics.

    Args:
        returns_df (pd.DataFrame): Asset or portfolio daily log returns.
        positions_df (pd.DataFrame, optional): Portfolio weights or positions (for strategy simulation).
        risk_free_rate (float, optional): Risk-free rate for Sharpe/Kappa calculations.
        objective_weights (dict, optional): Weights for performance metrics.

    Returns:
        pd.Series: Performance scores for each asset/strategy.
    """
    if objective_weights is None:
        objective_weights = {"sharpe": 1.0}  # Default: Optimize purely for Sharpe ratio

    # If positions_df is provided, compute strategy-level returns
    if positions_df is not None:
        pd.set_option("future.no_silent_downcasting", True)
        strategy_returns = (
            (positions_df * returns_df).sum(axis=1).fillna(0).infer_objects(copy=False)
        )
        assets = {"Portfolio": strategy_returns}  # Treat as a single asset
    else:
        assets = returns_df  # Individual assets

    metrics = {}
    for asset, asset_returns in assets.items():
        asset_returns = asset_returns.dropna()
        if asset_returns.empty:
            metrics[asset] = np.nan  # Return NaN instead of skipping
            continue

        # Compute individual metrics
        cumulative_return = (asset_returns + 1).prod() - 1  # Simple return formula
        sr = sharpe_ratio(asset_returns, risk_free_rate)
        kp = kappa_ratio(asset_returns, order=3) if "kappa" in objective_weights else 0
        omega = (
            omega_ratio(asset_returns, threshold=0)
            if "omega" in objective_weights
            else 0
        )
        cvar = (
            conditional_var(asset_returns, alpha=0.05)
            if "min_vol_tail" in objective_weights
            else 0
        )
        vol = asset_returns.std() if "min_vol_tail" in objective_weights else 0

        # Apply penalty for negative cumulative return
        penalty = 0
        if cumulative_return < 0:
            penalty = 2 * abs(
                cumulative_return
            )  # Extra penalty for negative performance

        # Weighted composite score
        composite = (
            objective_weights.get("cumulative_return", 0.0) * cumulative_return
            + objective_weights.get("sharpe", 1.0) * sr
            + objective_weights.get("kappa", 0.0) * kp
            + objective_weights.get("omega", 0.0) * omega
            - objective_weights.get("min_vol_tail", 0.0) * (cvar + vol)
            - penalty  # Apply additional penalty for negative returns
        )

        metrics[asset] = composite

    return pd.Series(metrics)


def get_objective_weights(objective: str) -> dict:
    """
    Returns the objective weight dictionary based on the given objective.

    Args:
        objective (str): The optimization objective. Choices:
                         ["min_vol_tail", "kappa", "sk_mix", "sharpe", "so_mix", "omega", "aggro"]

    Returns:
        dict: Objective weight dictionary for computing performance metrics.
    """
    objective_mappings = {
        "min_vol_tail": {
            "cumulative_return": 0.0,  # Ignore total return
            "min_vol_tail": 1.0,  # Fully minimize vol-tail-risk
            "sharpe": 0.0,  # Ignore Sharpe ratio
            "kappa": 0.0,  # Ignore Kappa ratio
            "omega": 0.0,  # Ignore Omega ratio
        },
        "kappa": {
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "kappa": 1.0,  # Fully maximize Kappa ratio
            "omega": 0.0,
            "min_vol_tail": 0.0,
        },
        "omega": {
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "kappa": 0.0,
            "omega": 1.0,  # Fully maximize Omega
            "min_vol_tail": 0.0,
        },
        "sk_mix": {
            "cumulative_return": 0.0,
            "sharpe": 0.5,  # Balanced risk-adjusted return
            "kappa": 0.5,  # Some tail-risk adjustment
            "omega": 0.0,
            "min_vol_tail": 0.0,
        },
        "so_mix": {
            "cumulative_return": 0.0,
            "sharpe": 0.5,  # Balanced risk-adjusted return
            "kappa": 0.0,
            "omega": 0.5,  # Adding Omega ratio to capture asymmetry
            "min_vol_tail": 0.0,
        },
         "sharpe": {
            "cumulative_return": 0.0,
            "sharpe": 1.0,  # Fully maximize Sharpe ratio
            "kappa": 0.0,
            "omega": 0.0,
            "min_vol_tail": 0.0,
        },
        "aggro": {
            "cumulative_return": 1 / 3,  # Prioritize raw return
            "sharpe": 1 / 3,
            "kappa": 1 / 3,
            "omega": 0.0,  # Omega is risk-adjusted, aggro purely seeks raw return
            "min_vol_tail": 0.0,
        },
         "yolo": {
            "cumulative_return": 0.5,  # Prioritize raw return
            "sharpe": 0.5,
            "kappa": 0.0,
            "omega": 0.0,  # Omega is risk-adjusted, aggro purely seeks raw return
            "min_vol_tail": 0.0,
        },
    }

    if objective not in objective_mappings:
        raise ValueError(
            f"Unknown objective: {objective}. Choose from {list(objective_mappings.keys())}"
        )

    return objective_mappings[objective]


def strategy_composite_score(
    metrics: Dict[str, float], objective_weights: Dict[str, float] = None
) -> float:
    """
    Combines performance metrics into a composite score using a configurable weight setup.

    Args:
        metrics (dict): Dictionary with keys like "cumulative_return", "sharpe", "kappa".
        objective_weights (dict, optional): Weights for each metric.

    Returns:
        float: Composite performance score.
    """
    if objective_weights is None:
        objective_weights = {"sharpe": 1.0}  # Default: Sharpe-only optimization

    score = sum(
        objective_weights.get(metric, 0) * value
        for metric, value in metrics.items()
        if metric in objective_weights
    )

    return score
