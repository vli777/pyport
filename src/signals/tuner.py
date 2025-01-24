import optuna

from signals.evaluate_signals import evaluate_signal_accuracy, simulate_strategy_returns
from signals.weighted_signals import calculate_weighted_signals


def objective(trial, signals, returns_df):
    """
    The Optuna objective function.
    - Suggest weights and decay parameters.
    - Use your `calculate_weighted_signals` function.
    - Return a combined F1 + returns score.
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

    # Evaluate F1 score
    accuracy_metrics = evaluate_signal_accuracy(
        weighted_signals, returns_df, threshold=0.0
    )
    f1 = accuracy_metrics["f1_score"]  # Extract only F1 score

    # Simulate strategy returns
    strategy_perf = simulate_strategy_returns(
        weighted_signals, returns_df, threshold=0.0
    )
    final_return = (
        strategy_perf["follow_all"].iloc[-1]
        if not strategy_perf["follow_all"].empty
        else 0.0
    )

    # Combine F1 and final returns into a single score
    score = 0.7 * f1 + 0.3 * final_return
    return score


def run_optuna_optimization(signals, returns_df, n_trials=50):
    """
    High-level function to run the optimization with Optuna.
    """
    # Create an Optuna study with direction="maximize"
    study = optuna.create_study(direction="maximize")

    # Define a wrapper so that objective has access to signals, returns_df
    def optuna_objective(trial):
        return objective(trial, signals, returns_df)

    # Optimize
    study.optimize(optuna_objective, n_trials=n_trials)

    # Return the best parameters and best score
    print("Best trial:")
    print("  Value: ", study.best_value)
    print("  Params: ", study.best_params)

    return study.best_params, study.best_value
