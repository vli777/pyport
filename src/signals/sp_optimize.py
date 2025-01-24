import numpy as np
from scipy.optimize import minimize, LinearConstraint


def sp_optimize(weighted_signals, signals_df, returns_df):
    # Initial guess from LightGBM feature importance
    initial_weights = np.array(list(weighted_signals.values()))
    # Bounds and constraints
    bounds = [(0.2, 1)] * len(initial_weights)
    constraints = [
        # Linear constraints can be added here
    ]
    # Optimize weights
    optimal_weights = minimize(
        objective,
        initial_weights,
        args=(signals_df, returns_df),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    return optimal_weights


def objective(weights, signals_df, asset_returns, reg_lambda=0.1):
    """
    Objective function to minimize: negative total return with L2 regularization.
    Args:
        weights (np.array): Weights for each signal.
        signals_df (pd.DataFrame): MultiIndex DataFrame of signals (levels: [signal, ticker]).
        asset_returns (pd.DataFrame): Single-level DataFrame of returns (columns: tickers).
        reg_lambda (float): Regularization parameter.
    Returns:
        float: Negative total return with regularization penalty.
    """
    # Replace NaNs in signals_df with 0
    signals_df = signals_df.fillna(0)

    # Compute weighted signals for each ticker
    weighted_signals = sum(
        weights[i] * signals_df.xs(signal, level=0, axis=1)
        for i, signal in enumerate(signals_df.columns.levels[0])
    )

    # Ensure alignment with asset_returns
    if not weighted_signals.columns.equals(asset_returns.columns):
        raise ValueError("Columns in signals_df and asset_returns do not match.")

    # Portfolio returns: element-wise product of weighted signals and returns
    portfolio_returns = (weighted_signals * asset_returns).sum(axis=1)

    # Total return: sum of portfolio returns
    total_return = portfolio_returns.sum()

    # L2 regularization penalty
    regularization_penalty = reg_lambda * np.sum(weights**2)

    return -(total_return - regularization_penalty)
