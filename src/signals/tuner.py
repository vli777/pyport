import optuna

from signals.evaluate_signals import (
    evaluate_signal_accuracy,
    process_multiindex_signals,
    simulate_strategy_returns,
)
from signals.weighted_signals import calculate_weighted_signals


def objective(trial, signals, returns_df):
    """
    The Optuna objective function for MultiIndex signals with bullish and bearish categories.
    - Suggest weights and decay parameters.
    - Process each category ('bullish', 'bearish') separately.
    - Return a combined score (F1 + returns).
    """
    # Suggest decay type (categorical: linear, exponential, or none)
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

    # Initialize scores for both categories
    combined_f1 = 0.0
    combined_returns = 0.0

    for category in ["bullish", "bearish"]:
        # Process signals for the current category
        category_signals = process_multiindex_signals(
            weighted_signals, category, threshold=0.0
        )

        # Evaluate F1 score for the category
        accuracy_metrics = evaluate_signal_accuracy(
            category_signals, returns_df, category, threshold=0.0
        )
        f1 = accuracy_metrics["f1_score"]

        # Simulate strategy returns for the category
        strategy_perf = simulate_strategy_returns(
            category_signals, returns_df, threshold=0.0
        )
        final_return = (
            strategy_perf["follow_all"].iloc[-1]
            if not strategy_perf["follow_all"].empty
            else 0.0
        )

        # Accumulate category scores (adjust weights if necessary)
        combined_f1 += 0.5 * f1  # Weight equally between bullish and bearish
        combined_returns += 0.5 * final_return

    # Combine F1 and final returns into a single score
    score = 0.7 * combined_f1 + 0.3 * combined_returns
    return score


def run_optuna_optimization(signals, returns_df, n_trials=50):
    """
    High-level function to run the optimization with Optuna.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, signals, returns_df), n_trials=n_trials
    )

    return study.best_params, study.best_value
