import optuna
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from reversion.zscore_multiplier import simulate_strategy


def filter_with_reversion(
    signals: Dict[str, Dict[str, List[str]]],
    windows: Dict[str, Dict[str, int]],
    symbols: List[str],
    returns_df: pd.DataFrame,
    n_trials: int = 50,
) -> List[str]:
    """
    Filter tickers using weighted mean reversion signals from multiple time scales with strict exclusion.
    Optimizes weights and inclusion_threshold using Optuna internally.

    Args:
        signals (dict): Dict of signals for each time scale, e.g.:
            {
                'daily': {'exclude': [...], 'include': [...]},
                'weekly': {'exclude': [...], 'include': [...]},
            }
        windows (dict): Dict of optimal windows for each time scale, e.g.:
            {
                'daily': { 'SPY': 20, ... }
                'weekly': { 'SPY': 3, ... }
            }
        symbols (List[str]): List of all ticker symbols.
        returns_df (pd.DataFrame): Log returns DataFrame with tickers as columns.
        n_trials (int): Number of Optuna optimization trials.

    Returns:
        List[str]: Final list of filtered tickers.
    """

    def apply_reversion_filter(
        signals: Dict[str, Dict[str, List[str]]],
        weights: Dict[str, Dict[str, float]],
        inclusion_threshold: float,
        exclusion_threshold: float,
    ) -> Dict[str, List[str]]:
        """
        Filter tickers based on weighted signals and inclusion/exclusion thresholds.

        Args:
            signals (dict): Mean reversion signals per time scale.
            weights (dict): Weights for each time scale and signal type.
                Example:
                {
                    'daily': {'exclude': 0.5, 'include': 0.7},
                    'weekly': {'exclude': 0.3, 'include': 0.5},
                }
            inclusion_threshold (float): Minimum cumulative inclusion score to include a ticker.
            exclusion_threshold (float): Minimum cumulative exclusion score to exclude a ticker.

        Returns:
            List[str]: Final list of filtered tickers.
        """
        # Initialize dictionaries for cumulative scores
        exclusion_scores = {ticker: 0.0 for ticker in symbols}
        inclusion_scores = {ticker: 0.0 for ticker in symbols}

        # Aggregate weighted exclusion scores
        for period in ["daily", "weekly"]:
            exclude_weight = weights.get(period, {}).get("exclude", 0.0)
            for ticker in signals.get(period, {}).get("exclude", []):
                exclusion_scores[ticker] += exclude_weight

        # Aggregate weighted inclusion scores
        for period in ["daily", "weekly"]:
            include_weight = weights.get(period, {}).get("include", 0.0)
            for ticker in signals.get(period, {}).get("include", []):
                inclusion_scores[ticker] += include_weight

        # Determine tickers to exclude based on exclusion_threshold
        tickers_to_exclude = {
            ticker
            for ticker, score in exclusion_scores.items()
            if score >= exclusion_threshold
        }

        # Determine tickers to include based on inclusion_threshold
        # Exclude tickers that have been marked for exclusion
        tickers_to_include = [
            ticker
            for ticker, score in inclusion_scores.items()
            if score >= inclusion_threshold and ticker not in tickers_to_exclude
        ]

        return tickers_to_include

    def get_combined_z_scores(
        returns_df: pd.DataFrame,
        windows: Dict[str, Dict[str, int]],
        filtered_tickers: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate Z-scores for each time scale and ticker using their respective window sizes.

        Args:
            returns_df (pd.DataFrame): Log returns DataFrame.
            windows (Dict[str, Dict[str, int]]): Optimized window sizes per time scale and ticker.
            filtered_tickers (List[str]): List of tickers to include.

        Returns:
            Dict[str, pd.DataFrame]: Z-score DataFrames per time scale.
        """
        z_scores = {}
        for period in ["daily", "weekly"]:
            z_scores[period] = pd.DataFrame(
                index=returns_df.index, columns=filtered_tickers
            )
            for ticker in filtered_tickers:
                window = windows[period].get(ticker, 20)  # Default to 20 if not found
                rolling_mean = (
                    returns_df[ticker].rolling(window=window, min_periods=1).mean()
                )
                rolling_std = (
                    returns_df[ticker].rolling(window=window, min_periods=1).std()
                )
                # Avoid division by zero
                z_scores[period][ticker] = (
                    returns_df[ticker] - rolling_mean
                ) / rolling_std.replace(0, np.nan)
        return z_scores

    def get_combined_thresholds(
        z_scores: Dict[str, pd.DataFrame], weights: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Calculate dynamic thresholds for each time scale and ticker.

        Args:
            z_scores (Dict[str, pd.DataFrame]): Z-score DataFrames per time scale.
            weights (Dict[str, Dict[str, float]]): Weights for each time scale and signal type.

        Returns:
            Dict[str, Dict[str, Tuple[float, float]]]: Thresholds per time scale and ticker.
        """
        thresholds = {}
        for period in ["daily", "weekly"]:
            thresholds[period] = {}
            for ticker in z_scores[period].columns:
                z_std = z_scores[period][ticker].std()
                # Handle cases where z_std is zero or NaN
                if np.isnan(z_std) or z_std == 0:
                    z_std = 1.0  # Default to 1 to avoid zero thresholds
                overbought_threshold = weights[period] * z_std
                oversold_threshold = -weights[period] * z_std
                thresholds[period][ticker] = (overbought_threshold, oversold_threshold)
        return thresholds

    def objective(trial) -> float:
        """
        Optuna objective function to optimize inclusion weights and threshold.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: Objective value to maximize (e.g., Sharpe ratio * cumulative return).
        """
        # Step 1: Suggest raw weights for each time scale and signal type
        # For each timeframe and signal type, suggest a weight
        weight_daily_exclude = trial.suggest_float("weight_daily_exclude", 0.0, 2.0)
        weight_daily_include = trial.suggest_float("weight_daily_include", 0.0, 2.0)
        weight_weekly_exclude = trial.suggest_float("weight_weekly_exclude", 0.0, 2.0)
        weight_weekly_include = trial.suggest_float("weight_weekly_include", 0.0, 2.0)

        # Organize weights in a nested dictionary
        weights = {
            "daily": {"exclude": weight_daily_exclude, "include": weight_daily_include},
            "weekly": {
                "exclude": weight_weekly_exclude,
                "include": weight_weekly_include,
            },
        }

        # Step 2: Suggest inclusion threshold
        inclusion_threshold = trial.suggest_float(
            "inclusion_threshold", 0.1, 2.0, step=0.05
        )
        exclusion_threshold = trial.suggest_float(
            "exclusion_threshold", 0.1, 2.0, step=0.05
        )

        # Step 3: Apply the weighted signals to filter tickers
        filtered_tickers = apply_reversion_filter(
            signals=signals,
            weights=weights,
            inclusion_threshold=inclusion_threshold,
            exclusion_threshold=exclusion_threshold,
        )

        # Step 4: Penalize if no tickers are selected
        if not filtered_tickers:
            return float("-inf")  # Avoid invalid portfolios

        # Step 5: Recompute Z-scores and dynamic thresholds using optimized window sizes
        z_scores = get_combined_z_scores(
            returns_df=returns_df, windows=windows, filtered_tickers=filtered_tickers
        )

        dynamic_thresholds = get_combined_thresholds(z_scores=z_scores, weights=weights)

        # Step 6: Simulate the strategy using the final include list
        strategy_returns, cumulative_return = simulate_strategy(
            returns_df=returns_df[filtered_tickers],
            dynamic_thresholds=dynamic_thresholds,
            z_scores_df=z_scores,
        )

        # Step 7: Calculate Sharpe ratio
        if strategy_returns.std() == 0 or np.isnan(strategy_returns.std()):
            return float("-inf")  # Penalize invalid Sharpe ratio

        sharpe_ratio = strategy_returns.mean() / strategy_returns.std()

        # Step 8: Define the objective value (e.g., maximize Sharpe ratio * cumulative return)
        objective_value = sharpe_ratio * cumulative_return

        return objective_value

    # Create and run the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Fetch the best parameters
    best_params = study.best_params
    best_weights = {
        "daily": {
            "exclude": best_params.get("weight_daily_exclude", 1.0),
            "include": best_params.get("weight_daily_include", 1.0),
        },
        "weekly": {
            "exclude": best_params.get("weight_weekly_exclude", 1.0),
            "include": best_params.get("weight_weekly_include", 1.0),
        },
    }
    best_inclusion_threshold = best_params.get("inclusion_threshold", 0.5)
    best_exclusion_threshold = best_params.get("exclusion_threshold", 0.5)

    print(f"Best weights: {best_weights}")
    print(f"Best inclusion threshold: {best_inclusion_threshold}")
    print(f"Best exclusion threshold: {best_exclusion_threshold}")

    # Apply the best weights to filter tickers
    final_filtered_tickers = apply_reversion_filter(
        signals=signals,
        weights=best_weights,
        inclusion_threshold=best_inclusion_threshold,
        exclusion_threshold=best_exclusion_threshold,
    )

    print(
        f"Final filtered tickers ({len(final_filtered_tickers)}): {final_filtered_tickers}"
    )

    return final_filtered_tickers
