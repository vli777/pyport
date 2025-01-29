from typing import Dict, List, Tuple
import optuna
import pandas as pd


def objective(trial, final_signals: pd.Series, returns_df: pd.DataFrame) -> float:
    """
    Use Optuna to optimize the inclusion/exclusion thresholds.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        final_signals (pd.DataFrame): Weighted time-series signals per ticker.
        returns_df (pd.DataFrame): Log returns DataFrame.

    Returns:
        float: Cumulative return based on optimized thresholds.
    """
    # Ensure `final_signals` has datetime index
    final_signals.index = pd.to_datetime(final_signals.index)

    # Search space for inclusion/exclusion thresholds (percentiles)
    include_threshold_pct = trial.suggest_float(
        "include_threshold_pct", 0.1, 0.4, step=0.05
    )
    exclude_threshold_pct = trial.suggest_float(
        "exclude_threshold_pct", 0.1, 0.4, step=0.05
    )

    # Initialize positions DataFrame
    positions = pd.DataFrame(0, index=returns_df.index, columns=returns_df.columns)

    for date in returns_df.index:
        # Ensure `date` is in datetime format
        date = pd.to_datetime(date)

        # Only use past data up to `date`
        if date not in final_signals.index:
            continue  # Skip if no signal for this date

        current_signals = final_signals.loc[date]  # Get signals for this specific date

        include_threshold = current_signals.quantile(1 - include_threshold_pct)
        exclude_threshold = current_signals.quantile(exclude_threshold_pct)

        include_tickers = current_signals[
            current_signals >= include_threshold
        ].index.tolist()
        exclude_tickers = current_signals[
            current_signals <= exclude_threshold
        ].index.tolist()

        # Ensure no ticker is in both
        include_tickers = list(set(include_tickers) - set(exclude_tickers))
        exclude_tickers = list(set(exclude_tickers) - set(include_tickers))

        positions.loc[date, include_tickers] = 1
        positions.loc[date, exclude_tickers] = -1

    # Simulate strategy
    _, cumulative_return = simulate_strategy(returns_df, positions)

    return cumulative_return  # Optuna maximizes this


def simulate_strategy(
    returns_df: pd.DataFrame, positions_df: pd.DataFrame
) -> Tuple[pd.Series, float]:
    """
    Simulates the strategy using positions and calculates cumulative return.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        positions_df (pd.DataFrame): Positions DataFrame with tickers as columns and dates as index.

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - strategy_returns (pd.Series): Daily returns of the strategy.
            - cumulative_return (float): Final cumulative return of the strategy.
    """
    # Calculate daily strategy returns using previous day's positions to avoid look-ahead bias
    strategy_returns = (positions_df.shift(1) * returns_df).sum(axis=1)

    # Calculate cumulative return
    cumulative_return = (
        strategy_returns + 1
    ).prod() - 1  # More accurate cumulative return

    return strategy_returns, cumulative_return


# Ensure final_signals and returns_df are defined before optimization
def find_optimal_inclusion_pct(
    final_signals: pd.Series, returns_df: pd.DataFrame, n_trials: int = 50
) -> Dict[str, float]:
    """
    Uses Optuna to optimize the inclusion/exclusion percentiles.

    Args:
        final_signals (pd.Series): Weighted signal scores per ticker.
        returns_df (pd.DataFrame): Log returns DataFrame.
        n_trials (int, optional): Number of trials for Optuna optimization. Defaults to 50.

    Returns:
        Dict[str, float]: A dictionary containing the best inclusion/exclusion percentiles.
    """
    # Create an Optuna study and optimize the thresholds
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, final_signals, returns_df), n_trials=n_trials
    )

    # Get the best thresholds
    best_include_pct = study.best_params["include_threshold_pct"]
    best_exclude_pct = study.best_params["exclude_threshold_pct"]

    print(
        f"✅ Optimal thresholds found: Include {best_include_pct}, Exclude {best_exclude_pct}"
    )

    return {
        "include_threshold_pct": best_include_pct,
        "exclude_threshold_pct": best_exclude_pct,
    }
