# file: src/core.py

from pathlib import Path
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from config import Config
from plotly_graphs import plot_graphs
from portfolio_optimization import run_optimization_and_save
from process_symbols import process_symbols
from result_output import output
from signals.reversion import apply_mean_reversion
from filters.anomaly_detection import remove_anomalous_stocks
from filters.decorrelation import filter_correlated_groups
from filters.filter_with_signals import filter_symbols_with_signals
from signals.generate_signals import generate_signals
from utils.caching_utils import cleanup_cache
from utils.data_utils import process_input_files
from utils.date_utils import calculate_start_end_dates
from utils.portfolio_utils import (
    calculate_performance_metrics,
    normalize_weights,
    stacked_output,
)


logger = logging.getLogger(__name__)


def run_pipeline(
    config: Config, symbols_override: Optional[List[str]] = None, run_local: bool = True
) -> Dict[str, Any]:
    """
    Orchestrates the data loading, optimization, and more.

    Args:
        config (Config): Configuration object parsed from YAML.
        symbols_override (Optional[List[str]]): Override for ticker symbols.
        run_local (bool): If True, show local plots and print logs to console.

    Returns:
        A dict (JSON-serializable) with final results or an empty dict if no data.
    """
    # Get ticker symbols
    if symbols_override:
        if not isinstance(symbols_override, list) or not all(
            isinstance(s, str) for s in symbols_override
        ):
            raise ValueError("symbols_override must be a list of strings.")
        logger.info(f"Received override of symbols: {symbols_override}")
        all_symbols = symbols_override
    else:
        # Use the watchlists from config
        watchlist_files = [
            Path(config.input_files_dir) / file for file in config.input_files
        ]
        all_symbols = process_input_files(watchlist_files)

    if not all_symbols:
        logger.warning("No symbols found. Aborting pipeline.")
        return {}

    # Main pipeline logic
    stack: Dict[str, Any] = {}
    dfs: Dict[str, Any] = {}
    filtered_times = [k for k in config.models if config.models[k]]
    sorted_times = sorted(filtered_times, reverse=True)

    # Calculate start and end dates for the longest time period
    start_date_long, end_date_long = calculate_start_end_dates(sorted_times[0])

    # Load data for the longest period using all symbols
    df_all = process_symbols(
        symbols=all_symbols,
        start_date=start_date_long,
        end_date=end_date_long,
        data_path=Path(config.data_dir),
        download=config.download,
    )

    # Calculate daily returns for the longest period
    df_adj = df_all.xs("Adj Close", level=1, axis=1)
    returns_df = df_adj.pct_change().dropna()

    # Remove anomalous stocks
    returns_df = remove_anomalous_stocks(
        returns_df,
        threshold=config.anomaly_detection_deviation_threshold,
        plot=config.plot_anomalies,
    )

    # Calculate performance metrics
    performance_df = calculate_performance_metrics(
        returns_df, risk_free_rate=config.risk_free_rate
    )

    # Filter overbought, oversold
    filtered_symbols = filter_symbols_with_signals(
        price_df=df_all,
        returns_df=returns_df,
        generate_signals_fn=generate_signals,
        mean_reversion_fn=apply_mean_reversion,
        config=config,
    )

    # Filter correlated groups based on performance metrics
    filtered_returns_df = returns_df[filtered_symbols]
    filtered_decorrelated_symbols = filter_correlated_groups(
        returns_df=filtered_returns_df,
        performance_df=performance_df,
        sharpe_threshold=0.005,
        correlation_threshold=config.correlation_threshold,
        plot=config.plot_clustering,
    )

    # Combine filtered symbols
    print(f"Symbols for further optimization: {filtered_decorrelated_symbols}")

    dfs.update({"data": df_adj, "start": start_date_long, "end": end_date_long})

    for years in sorted_times:
        start_date, end_date = calculate_start_end_dates(years)
        if config.test_mode:
            logger.info(f"Time period: {years}, symbols: {all_symbols}")

        # Load data
        df_adj = df_adj.loc[start_date:end_date].copy()

        # Update min/max dates in the dfs dict
        dfs["start"] = min(dfs["start"], start_date)
        dfs["end"] = max(dfs["end"], end_date)

        # Optional: If test_mode is on, store a CSV of the full data
        if config.test_mode:
            df_adj.to_csv("full_df.csv")
            df_adj = df_adj.head(int(len(df_adj) * config.test_data_visible_pct))

        # Run optimization
        run_optimization_and_save(
            df_adj,
            config,
            start_date,
            end_date,
            filtered_decorrelated_symbols,
            stack,
            years,
        )

    if not stack:
        logger.warning("No optimization results found.")
        return {}

    # Post-processing
    for key, value in stack.items():
        if isinstance(value, pd.Series):
            print(key, "stored as pd")
            stack[key] = value.to_dict()

    avg = stacked_output(stack)
    if not avg:
        logger.warning("No valid averaged weights found. Skipping further processing.")

    sorted_avg = dict(sorted(avg.items(), key=lambda item: item[1]))
    normalized_avg = normalize_weights(sorted_avg, config.min_weight)

    valid_models = [v for v in config.models.values() if v]
    combined_models = ", ".join(sorted(set(sum(valid_models, []))))
    combined_input_files = ", ".join(config.input_files)

    daily_returns, cumulative_returns = output(
        data=dfs["data"],
        allocation_weights=normalized_avg,
        inputs=combined_input_files,
        start_date=dfs["start"],
        end_date=dfs["end"],
        optimization_model=combined_models,
        time_period=sorted_times[0],
        minimum_weight=config.min_weight,
        max_size=config.portfolio_max_size,
        config=config,
    )

    # Sort symbols by allocation weights (descending)
    sorted_symbols = [
        symbol
        for symbol, _ in sorted(
            zip(daily_returns.columns, normalized_avg),
            key=lambda x: x[1],  # Sort by weight
            reverse=True,  # Descending order
        )
    ]

    if run_local:
        plot_graphs(
            daily_returns,
            cumulative_returns,
            config,
            symbols=daily_returns.columns.tolist(),
        )

    cleanup_cache("cache")

    return {
        "start_date": str(dfs["start"]),
        "end_date": str(dfs["end"]),
        "models": combined_models,
        "symbols": sorted_symbols,
        "normalized_avg": normalized_avg,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
    }
