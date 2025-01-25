import sys
import numpy as np

from signals.technicals import (
    calculate_macd,
    calculate_rainbow_stoch,
    calculate_stoch_buy_persistence,
    generate_macd_crossovers,
    generate_macd_preemptive_signals,
    calculate_adx,
    generate_adx_signals,
    generate_convergence_signals,
)
from signals.calculate_weighted_signals import (
    calculate_weighted_signals,
    verify_ticker_consistency,
    verify_weighted_signals,
)
from signals.signals_filter import filter_signals_by_threshold
from signals.optimize_signal_weights import run_optuna_optimization
from signals.signal_threshold import plot_threshold_metrics, analyze_thresholds


def generate_signals(
    price_df, returns_df, plot=False, buy_threshold=1.0, sell_threshold=1.0
):
    """
    Generate all technical signals and return a consolidated DataFrame.
    Only uses the latest date's signals without aggregation.

    Returns:
        consolidated_signals (pd.DataFrame): DataFrame containing all signal weights and indicator values per ticker.
    """
    # Generate MACD signals
    macd_line_df, macd_signal_df, macd_hist_df = calculate_macd(price_df)
    bullish_crossover, bearish_crossover = generate_macd_crossovers(
        macd_line_df, macd_signal_df
    )
    preemptive_bullish, preemptive_bearish = generate_macd_preemptive_signals(
        macd_line_df
    )

    # Calculate ADX and generate ADX trend signals
    adx_df = calculate_adx(price_df)
    adx_signals = generate_adx_signals(adx_df)

    # Calculate Stochastic Oscillator and generate convergence signals
    stoch_ks = calculate_rainbow_stoch(price_df)
    stoch_buy_signal, stoch_sell_signal = generate_convergence_signals(stoch_ks)

    # Calculate Stochastic persistence based on LP signal weight testing
    stoch_buy_persistence = calculate_stoch_buy_persistence(stoch_buy_signal)

    signals = {
        "stoch_buy": stoch_buy_signal,
        "stoch_sell": stoch_sell_signal,
        "bullish_crossover": bullish_crossover,
        "bearish_crossover": bearish_crossover,
        "preemptive_bullish": preemptive_bullish,
        "preemptive_bearish": preemptive_bearish,
        "adx_support": adx_signals,
        "stoch_buy_persistence": stoch_buy_persistence,
    }

    # signals: dict of {signal_name: DataFrame}, each DataFrame is date x ticker
    # returns_df: DataFrame, date x ticker
    # signal weight optimization
    best_signal_weights, best_score = run_optuna_optimization(
        signals,
        returns_df,
        n_trials=50,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    print(f"\nbest weights: {best_signal_weights}")
    print(f"best score {best_score}\n")

    # Use the best parameters to calculate final weighted signals
    optimal_signal_weights = {
        name.replace("weight_", ""): w
        for name, w in best_signal_weights.items()
        if name.startswith("weight_")
    }

    final_weighted_signals = calculate_weighted_signals(
        signals=signals,
        signal_weights=optimal_signal_weights,
        days=7,
        weight_decay="exponential",
    )

    # Verify the integrity of weighted_signals
    verify_weighted_signals(final_weighted_signals)
    verify_ticker_consistency(final_weighted_signals, returns_df)

    print(f"\nfinal weighted signals: {final_weighted_signals}")

    if plot:
        # Flatten all signal values
        all_values = final_weighted_signals.values.flatten()
        all_values = all_values[~np.isnan(all_values)]

        # Generate thresholds
        thresholds = np.linspace(min(all_values), max(all_values), 20)

        # Analyze thresholds for bullish signals
        bullish_metrics = analyze_thresholds(
            final_weighted_signals, returns_df, thresholds, "bullish"
        )

        # Analyze thresholds for bearish signals
        bearish_metrics = analyze_thresholds(
            final_weighted_signals, returns_df, thresholds, "bearish"
        )

        # Plot signal threshold metric
        plot_threshold_metrics(
            thresholds=thresholds,
            bullish_metrics=bullish_metrics,
            bearish_metrics=bearish_metrics,
            buy_threshold=1.0,
            sell_threshold=1.0,
        )

    # Filter based on thresholds
    buy_signal_tickers, sell_signal_tickers = filter_signals_by_threshold(
        weighted_signals=final_weighted_signals,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    return buy_signal_tickers, sell_signal_tickers
