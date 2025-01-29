from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import optuna

from anomaly.plot_anomalies import plot_anomalies
from anomaly.kalman_filter import apply_kalman_filter


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]],
    plot: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Removes stocks with anomalous returns based on the Kalman filter.
    Optionally plots anomalies for all flagged stocks in a paginated 6x6 grid.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns.
        weight_dict (dict): Dictionary with optional objective weights, e.g. { sortino: 0.8, stability 0.2 }
        plot (bool): If True, anomalies will be plotted in a paginated grid.

    Returns:
        pd.DataFrame: Filtered DataFrame with anomalous stocks removed.
    """
    threshold = optimize_kalman_threshold(returns_df, weight_dict)

    anomalous_cols = []

    # Dictionaries to store data for plotting if needed
    returns_data = {}
    anomaly_flags_data = {}

    for stock in returns_df.columns:
        returns_series = returns_df[stock].dropna()
        if returns_series.empty:
            print(f"Warning: No data for stock {stock}. Skipping.")
            continue

        anomaly_flags = apply_kalman_filter(returns_series, threshold=threshold)

        # If anomalies found for the stock
        if anomaly_flags.any():
            anomalous_cols.append(stock)
            # Store data for plotting if plot is True
            if plot:
                returns_data[stock] = returns_series
                anomaly_flags_data[stock] = anomaly_flags

    print(
        f"Removing {len(anomalous_cols)} stocks with Kalman anomalies: {anomalous_cols}"
    )

    # If plotting is requested and there are stocks with anomalies, plot them
    if plot and returns_data:
        # Use the list of anomalous stocks for plotting
        plot_anomalies(
            stocks=anomalous_cols,
            returns_data=returns_data,
            anomaly_flags_data=anomaly_flags_data,
            stocks_per_page=36,
        )

    # Return the DataFrame with anomalous stocks removed
    filtered_df = returns_df.drop(columns=anomalous_cols)
    return filtered_df, anomalous_cols


def objective(trial, returns_df: pd.DataFrame, weight_dict: dict = None) -> float:
    """
    Optimize Kalman filter threshold using a combined objective function.

    Balances:
    - Sortino Ratio (risk-adjusted return)
    - Stability Penalty (rolling volatility to avoid meme stocks)

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        returns_df (pd.DataFrame): DataFrame of daily returns.
        weight_dict (dict): Dictionary with weight settings. Defaults to {'sortino': 0.8, 'stability': 0.2}.

    Returns:
        float: Composite score (higher is better).
    """
    # Default weights if not provided
    if weight_dict is None:
        weight_dict = {"sortino": 0.8, "stability": 0.2}

    # Ensure both weights sum to 1 (normalize if necessary)
    total_weight = sum(weight_dict.values())
    weight_sortino = weight_dict.get("sortino", 0.8) / total_weight
    weight_stability = weight_dict.get("stability", 0.2) / total_weight

    threshold = trial.suggest_float("threshold", 5.0, 10.0, step=0.5)

    # Remove anomalous stocks
    filtered_df, _ = remove_anomalous_stocks(
        returns_df, threshold=threshold, plot=False
    )

    if filtered_df.empty:
        return -np.inf  # Penalize empty selections

    # Portfolio return
    portfolio_return = filtered_df.mean(axis=1).mean()

    # Calculate downside deviation (only negative returns)
    negative_returns = filtered_df.mean(axis=1)[filtered_df.mean(axis=1) < 0]
    downside_risk = negative_returns.std()

    # Avoid division by zero
    if downside_risk == 0:
        return -np.inf  # Penalize cases where no downside risk is captured

    # Sortino Ratio (risk-adjusted return)
    sortino_ratio = portfolio_return / downside_risk

    # Stability Penalty (rolling volatility)
    rolling_volatility = filtered_df.mean(axis=1).rolling(window=30).std().mean()

    # Avoid division by zero
    if rolling_volatility == 0:
        rolling_volatility = 1e-6  # Prevent divide by zero

    stability_penalty = (
        -rolling_volatility * 0.1
    )  # Reduce weight of excessive volatility

    # Weighted sum of Sortino Ratio and Stability Penalty
    composite_score = (weight_sortino * sortino_ratio) + (
        weight_stability * stability_penalty
    )

    return composite_score


def optimize_kalman_threshold(
    returns_df: pd.DataFrame, n_trials: int = 50, weight_dict: dict = None
):
    """
    Optimize the Kalman filter threshold using a multi-objective approach.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int): Number of Optuna trials.
        weight_dict (dict): Dictionary with weight settings (e.g., {'sortino': 0.8, 'stability': 0.2}).

    Returns:
        float: Best threshold value.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, returns_df, weight_dict),
        n_trials=n_trials,
        n_jobs=-1,
    )

    best_threshold = study.best_trial.params["threshold"]
    print(f"Best Kalman threshold found: {best_threshold} with weights {weight_dict}")
    return best_threshold
