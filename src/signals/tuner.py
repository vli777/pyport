import hashlib
import os
import pickle
from typing import Any, Dict, Optional, Tuple
import optuna
import pandas as pd

from signals.evaluate_signals import (
    evaluate_signal_accuracy,
    process_multiindex_signals,
    simulate_strategy_returns,
)
from signals.weighted_signals import calculate_weighted_signals


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
    save_path: Optional[str] = None,
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
    # Generate a unique hash for the current set of signals
    signals_hash = generate_signals_hash(signals)

    # Define the cache file path
    if save_path is None:
        # Default to current directory if save_path is not provided
        save_path = os.getcwd()
    cache_file = os.path.join(save_path, f"optuna_result_{signals_hash}.pkl")

    # Attempt to load cached results
    cached_result = load_from_cache(cache_file)
    if cached_result:
        best_signal_weights = cached_result["best_signal_weights"]
        best_score = cached_result["best_score"]
        return best_signal_weights, best_score

    # If no cache is found, proceed with optimization
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

    # Save the results to cache
    save_to_cache(cache_file, best_signal_weights, best_score)

    return best_signal_weights, best_score


def generate_signals_hash(signals: Dict[str, Any]) -> str:
    """
    Generates a SHA256 hash based on the sorted keys of the signals dictionary.

    Args:
        signals (Dict[str, Any]): Dictionary of signals.

    Returns:
        str: Hexadecimal SHA256 hash string.
    """
    sorted_keys = sorted(signals.keys())
    keys_str = ",".join(sorted_keys)
    hash_object = hashlib.sha256(keys_str.encode("utf-8"))
    return hash_object.hexdigest()


def save_to_cache(
    file_path: str, best_signal_weights: Dict[str, float], best_score: float
) -> None:
    """
    Saves the optimization results to a pickle file.

    Args:
        file_path (str): Full path to the pickle file.
        best_signal_weights (Dict[str, float]): Best signal weights found by optimization.
        best_score (float): Best score achieved during optimization.
    """
    with open(file_path, "wb") as f:
        pickle.dump(
            {"best_signal_weights": best_signal_weights, "best_score": best_score}, f
        )
    print(f"Saved optimization results to cache: {file_path}")


def load_from_cache(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads the optimization results from a pickle file if available.

    Args:
        file_path (str): Full path to the pickle file.

    Returns:
        Optional[Dict[str, Any]]: Loaded optimization results or None if not found.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            cached_result = pickle.load(f)
        print(f"Loaded optimization results from cache: {file_path}")
        return cached_result
    else:
        return None
