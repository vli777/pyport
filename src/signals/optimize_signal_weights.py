from typing import Any, Dict, Optional, Tuple
import optuna
import pandas as pd

from signals.evaluate_signal_metrics import (
    evaluate_signal_accuracy,
    process_multiindex_signals,
    simulate_strategy_returns,
)
from signals.calculate_weighted_signals import calculate_weighted_signals


def objective(
    trial,
    signals: Dict[str, pd.DataFrame],
    returns_df: pd.DataFrame,
    buy_threshold: float = 5.0,
    sell_threshold: float = 3.5,
) -> float:
    """
    Optuna objective function to optimize signal weights.

    Args:
        trial: Optuna trial object.
        signals (Dict[str, pd.DataFrame]): Dictionary of signal DataFrames.
        returns_df (pd.DataFrame): DataFrame of actual stock returns.
        buy_threshold (float, optional): Threshold for bullish signals.
        sell_threshold (float, optional): Threshold for bearish signals.

    Returns:
        float: Combined score based on F1 and returns.
    """
    # Suggest decay type
    decay = trial.suggest_categorical("decay", ["linear", "exponential", None])

    # Suggest weights for each signal
    signal_weights = {
        signal_name: trial.suggest_float(f"weight_{signal_name}", 0.1, 1.0)
        for signal_name in signals.keys()
    }

    # Calculate weighted signals
    weighted_signals = calculate_weighted_signals(
        signals=signals,
        signal_weights=signal_weights,
        days=7,
        weight_decay=decay,
    )

    # Process bullish signals
    buy_signals = process_multiindex_signals(weighted_signals, "bullish", buy_threshold)

    # Process bearish signals
    sell_signals = process_multiindex_signals(
        weighted_signals, "bearish", sell_threshold
    )

    # Check if there are any buy or sell signals
    if buy_signals.empty and sell_signals.empty:
        print("Warning: No buy or sell signals found.")
        return 0.0  # Or another appropriate default score

    # Evaluate metrics for bullish signals
    if not buy_signals.empty:
        bullish_metrics = evaluate_signal_accuracy(
            category_signals=buy_signals,
            returns_df=returns_df,
            threshold=buy_threshold,
        )
    else:
        bullish_metrics = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # Evaluate metrics for bearish signals
    if not sell_signals.empty:
        bearish_metrics = evaluate_signal_accuracy(
            category_signals=sell_signals,
            returns_df=returns_df,
            threshold=sell_threshold,
        )
    else:
        bearish_metrics = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # Combine F1 scores
    combined_f1 = 0.5 * bullish_metrics["f1_score"] + 0.5 * bearish_metrics["f1_score"]

    # Simulate strategies
    strategy_perf = simulate_strategy_returns(
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        returns_df=returns_df,
    )

    # Extract final cumulative returns
    final_return_follow_all = (
        strategy_perf["follow_all"].iloc[-1]
        if not strategy_perf["follow_all"].empty
        else 0.0
    )
    final_return_avoid_bearish = (
        strategy_perf["avoid_bearish"].iloc[-1]
        if not strategy_perf["avoid_bearish"].empty
        else 0.0
    )
    final_return_partial = (
        strategy_perf["partial_adherence"].iloc[-1]
        if not strategy_perf["partial_adherence"].empty
        else 0.0
    )

    # Combine returns with appropriate weights
    combined_returns = (
        final_return_follow_all + final_return_avoid_bearish + final_return_partial
    ) / 3

    # Combine F1 and returns into a single score
    score = 0.7 * combined_f1 + 0.3 * combined_returns

    return score


def run_optuna_optimization(
    signals: Dict[str, pd.DataFrame],
    returns_df: pd.DataFrame,
    n_trials: int = 50,
    buy_threshold: float = 0.0,
    sell_threshold: float = 0.0,
) -> Tuple[Dict[str, float], float]:
    """
    High-level function to run the optimization with Optuna, utilizing caching.

    Args:
        signals (Dict[str, pd.DataFrame]): Dictionary of signal DataFrames.
        returns_df (pd.DataFrame): DataFrame of actual stock returns.
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        buy_threshold (float, optional): Threshold for bullish signals. Defaults to 0.0.
        sell_threshold (float, optional): Threshold for bearish signals. Defaults to 0.0.
        save_path (Optional[str], optional): Directory path to save/load cache files. Defaults to None.

    Returns:
        Tuple[Dict[str, float], float]: Best signal weights and corresponding best score.
    """
    study = optuna.create_study(direction="maximize")

    # Define the objective function with necessary parameters
    def optuna_objective(trial):
        return objective(
            trial=trial,
            signals=signals,
            returns_df=returns_df,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

    # Run the optimization
    study.optimize(optuna_objective, n_trials=n_trials)

    # Extract the best weights and score
    best_weights = study.best_params
    # Remove the 'weight_' prefix from keys
    best_signal_weights = {k.replace("weight_", ""): v for k, v in best_weights.items()}
    best_score = study.best_value

    return best_signal_weights, best_score
