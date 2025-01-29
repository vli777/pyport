# file: src/core.py

from pathlib import Path
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import Config
from plotly_graphs import plot_graphs
from portfolio_optimization import run_optimization_and_save
from process_symbols import process_symbols
from result_output import output
from anomaly.anomaly_detection import remove_anomalous_stocks
from correlation.decorrelation import filter_correlated_groups
from correlation.optimize_correlation import optimize_correlation_threshold
from reversion.multiscale_reversion import apply_mean_reversion_multiscale
from reversion.optimize_timescale_weights import find_optimal_weights
from reversion.recommendation import generate_reversion_recommendations
from reversion.optimize_inclusion import find_optimal_inclusion_pct
from boxplot import generate_boxplot_data
from utils.performance_metrics import calculate_performance_metrics
from utils.caching_utils import cleanup_cache
from utils.data_utils import process_input_files
from utils.date_utils import calculate_start_end_dates
from utils.portfolio_utils import (
    normalize_weights,
    stacked_output,
)


logger = logging.getLogger(__name__)


def run_pipeline(
    config: Config,
    symbols_override: Optional[List[str]] = None,
    run_local: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrates the data loading, optimization, and analysis pipeline.

    Args:
        config (Config): Configuration object parsed from YAML.
        symbols_override (Optional[List[str]]): Override for ticker symbols.
        run_local (bool): If True, display local plots and log to console.

    Returns:
        Dict[str, Any]: JSON-serializable dictionary containing final results or empty if no data.
    """

    def validate_symbols_override(overrides: List[str]) -> None:
        if not all(isinstance(symbol, str) for symbol in overrides):
            raise ValueError("All elements in symbols_override must be strings.")
        logger.info(f"Symbols overridden: {overrides}")

    def load_symbols() -> List[str]:
        if symbols_override:
            validate_symbols_override(symbols_override)
            return symbols_override
        watchlist_paths = [
            Path(config.input_files_dir) / file for file in config.input_files
        ]
        symbols = process_input_files(watchlist_paths)
        logger.info(f"Loaded symbols from watchlists: {symbols}")
        return symbols

    def load_data(
        all_symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        logger.debug(
            f"Loading data for symbols: {all_symbols} from {start_date} to {end_date}"
        )
        try:
            data = process_symbols(
                symbols=all_symbols,
                start_date=start_date,
                end_date=end_date,
                data_path=Path(config.data_dir),
                download=config.download,
            )
            if data.empty:
                logger.warning("Loaded data is empty.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from a multiindex DataFrame with adjusted close prices.

        Args:
            df (pd.DataFrame): Multiindex DataFrame with adjusted close prices under the level "Adj Close".

        Returns:
            pd.DataFrame: DataFrame containing daily returns for each stock.
        """
        try:
            # Extract only 'Adj Close' level
            adj_close = df.xs("Adj Close", level=1, axis=1)

            # Calculate daily returns while preserving different stock histories
            returns = adj_close.pct_change()

            # Fill only leading NaNs for stocks with different start dates
            returns = returns.ffill().dropna(how="all")

            logger.debug("Calculated daily returns.")
            return returns

        except KeyError as e:
            logger.error(
                f"Adjusted close prices not found in the DataFrame. Error: {e}"
            )
            raise

    def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess the input DataFrame to calculate returns and optionally remove anomalous stocks.

        Args:
            df (pd.DataFrame): Multiindex DataFrame with adjusted close prices.
            config: Configuration object with settings like anomaly detection threshold and plotting options.

        Returns:
            pd.DataFrame: Processed DataFrame with daily returns and optional anomaly filtering applied.
        """
        returns = calculate_returns(df)

        removed_symbols = []
        if config.use_anomaly_filter:
            logger.debug("Applying anomaly filter.")
            returns, removed_symbols = remove_anomalous_stocks(
                returns,
                # weight_dict # placeholder for configurable weights based on risk preference
                plot=config.plot_anomalies,
            )
        else:
            logger.debug("Skipping anomaly filter.")

        return returns, removed_symbols

    def filter_symbols(returns_df: pd.DataFrame, config: Config) -> List[str]:
        """
        Apply mean reversion and decorrelation filters to return valid symbols.
        Falls back to the original returns_df columns if filtering results in an empty list.
        """
        original_symbols = list(returns_df.columns)  # Preserve original symbols

        # Apply mean reversion signals (if enabled)
        if config.use_reversion_filter:
            # Generate Reversion Signals
            reversion_signals = apply_mean_reversion_multiscale(
                returns_df, n_trials=50, n_jobs=-1, plot=config.plot_reversion_threshold
            )
            print("Reversion Signals Generated.")

            # Optimize Weights
            optimal_weights = find_optimal_weights(
                reversion_signals, returns_df, n_trials=50
            )
            print(f"Optimal Weights: {optimal_weights}")

            # Generate Initial Recommendations
            final_recommendations = generate_reversion_recommendations(
                reversion_signals, optimal_weights, include_pct=0.2, exclude_pct=0.2
            )
            print("Initial Tickers to include:", final_recommendations["include"])
            print("Initial Tickers to exclude:", final_recommendations["exclude"])

            # Align signal data properly
            daily_signals = pd.DataFrame.from_dict(
                reversion_signals["daily"], orient="index"
            ).T
            weekly_signals = pd.DataFrame.from_dict(
                reversion_signals["weekly"], orient="index"
            ).T

            # Use available trading history instead of forcing all stocks into the same range
            all_dates = (
                returns_df.index
            )  # Use returns_df index as the reference trading calendar
            daily_signals = daily_signals.reindex(all_dates).fillna(0)
            weekly_signals = weekly_signals.reindex(all_dates).fillna(0)

            # Compute weighted signal strength
            weight_daily = optimal_weights.get("weight_daily", 0.5)
            weight_weekly = 1.0 - weight_daily
            final_signals = (
                weight_daily * daily_signals + weight_weekly * weekly_signals
            )

            # Optimize Inclusion/Exclusion Thresholds
            optimal_inclusion_thresholds = find_optimal_inclusion_pct(
                final_signals, returns_df, n_trials=50
            )
            print(f"✅ Optimal Inclusion Thresholds: {optimal_inclusion_thresholds}")

            # Generate Final Recommendations with optimized thresholds
            reversion_recommendations = generate_reversion_recommendations(
                reversion_signals,
                optimal_weights,
                include_pct=optimal_inclusion_thresholds.get(
                    "include_threshold_pct", 0.2
                ),
                exclude_pct=optimal_inclusion_thresholds.get(
                    "exclude_threshold_pct", 0.2
                ),
            )

            # Modify Trading Universe
            include_tickers = set(reversion_recommendations["include"])
            exclude_tickers = set(reversion_recommendations["exclude"])

            # Ensure exclusions are applied and inclusions are added
            filtered_symbols = (
                set(original_symbols) - exclude_tickers
            ) | include_tickers
            filtered_symbols = sorted(filtered_symbols)  # Ensure consistency

        else:
            filtered_symbols = original_symbols

        # Ensure final symbols exist in returns_df (not just original_symbols)
        valid_symbols = [
            symbol for symbol in filtered_symbols if symbol in returns_df.columns
        ]

        if len(valid_symbols) < len(filtered_symbols):
            missing_symbols = set(filtered_symbols) - set(valid_symbols)
            logger.warning(f"Filtered symbols not in returns_df: {missing_symbols}")

        # Apply correlation filter (if enabled)
        if config.use_correlation_filter and valid_symbols:
            try:
                # Calculate performance metrics
                performance_metrics = calculate_performance_metrics(
                    returns_df[valid_symbols], config.risk_free_rate
                )

                # Load market returns
                market_returns = calculate_returns(
                    load_data(
                        all_symbols=["SPY"], start_date=start_long, end_date=end_long
                    )
                )

                # Optimize correlation threshold
                best_params, best_value = optimize_correlation_threshold(
                    returns_df=returns_df[valid_symbols],
                    performance_df=performance_metrics,
                    market_returns=market_returns,
                    risk_free_rate=config.risk_free_rate,
                )

                # Filter decorrelated tickers
                decorrelated_tickers = filter_correlated_groups(
                    returns_df=returns_df[valid_symbols],
                    performance_df=performance_metrics,
                    correlation_threshold=best_params["correlation_threshold"],
                    sharpe_threshold=0.005,
                    plot=config.plot_clustering,
                    top_n=config.top_n_candidates,
                )

                valid_symbols = [
                    symbol for symbol in valid_symbols if symbol in decorrelated_tickers
                ]

                logger.info(
                    f"Optimized correlation threshold: {best_params['correlation_threshold']:.4f}"
                )

            except Exception as e:
                logger.error(f"Correlation threshold optimization failed: {e}")
                # Fall back to original symbols if optimization fails
                valid_symbols = original_symbols

        # Ensure a non-empty list of symbols is returned
        if not valid_symbols:
            logger.warning(
                "No valid symbols after filtering. Returning original symbols."
            )
            valid_symbols = original_symbols

        return valid_symbols

    def perform_post_processing(stack_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform post-processing on the stack data to calculate normalized weights.

        Args:
            stack_data (Dict[str, Any]): The stack data containing optimization results.

        Returns:
            Dict[str, Any]: Normalized weights as a dictionary.
        """
        # Convert pd.Series to dictionaries if necessary
        processed_stack = {
            key: (value.to_dict() if isinstance(value, pd.Series) else value)
            for key, value in stack_data.items()
        }

        # Compute averaged weights
        average_weights = stacked_output(processed_stack)
        if not average_weights:
            logger.warning(
                "No valid averaged weights found. Skipping further processing."
            )
            return {}

        # Sort weights in descending order
        sorted_weights = dict(
            sorted(average_weights.items(), key=lambda item: item[1], reverse=True)
        )

        # Normalize weights and convert to a dictionary if necessary
        normalized_weights = normalize_weights(sorted_weights, config.min_weight)

        # Ensure output is a dictionary
        if isinstance(normalized_weights, pd.Series):
            normalized_weights = normalized_weights.to_dict()

        return normalized_weights

    # Step 1: Load and validate symbols
    try:
        all_symbols = load_symbols()
        if not all_symbols:
            logger.warning("No symbols found. Aborting pipeline.")
            return {}
    except ValueError as e:
        logger.error(f"Symbol override validation failed: {e}")
        return {}

    # Step 2: Initialize structures
    stack: Dict[str, Any] = {}
    dfs: Dict[str, Any] = {}
    active_models = [k for k, v in config.models.items() if v]
    sorted_time_periods = sorted(active_models, reverse=True)

    # Step 3: Determine date range based on the longest period
    longest_period = sorted_time_periods[0]
    start_long, end_long = calculate_start_end_dates(longest_period)
    dfs["start"] = start_long
    dfs["end"] = end_long

    # Step 4: Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df_all = load_data(all_symbols, start_long, end_long)

    # Ensure we use the **largest valid date range** for returns_df
    all_dates = df_all.index  # Keep full range before filtering
    returns_df, removed_anomalous = preprocess_data(df_all)

    # Log removed anomalous symbols
    if removed_anomalous:
        logger.info(f"Anomalous symbols removed: {removed_anomalous}")

    # Step 5: Filter symbols based on mean reversion only if required
    logger.info("Filtering symbols...")
    valid_symbols = filter_symbols(returns_df, config)

    # Ensure `valid_symbols` aligns with available trading history
    valid_symbols = [sym for sym in valid_symbols if sym in returns_df.columns]

    if not valid_symbols:
        logger.warning("No valid symbols remain after filtering. Aborting pipeline.")
        return {}

    logger.info(f"Symbols selected for optimization: {valid_symbols}")

    try:
        dfs["data"] = df_all.xs("Adj Close", level=1, axis=1)[valid_symbols]
        logger.debug(f"dfs['data'] shape: {dfs['data'].shape}")
    except KeyError as e:
        logger.error(f"Error slicing df_all with filtered_decorrelated: {e}")
        raise

    # Step 6: Iterate through each time period and perform optimization
    logger.info("Running optimization...")
    for period in sorted_time_periods:
        start, end = calculate_start_end_dates(period)
        logger.debug(f"Processing period: {period} from {start} to {end}")

        # Preserve full available date range
        df_period = df_all.loc[start:end].copy()

        # Flatten df_period for optimization
        if isinstance(
            df_period.columns, pd.MultiIndex
        ) and "Adj Close" in df_period.columns.get_level_values(1):
            try:
                df_period = df_period.xs("Adj Close", level=1, axis=1)

                # Ensure stocks with shorter histories remain in the dataset
                all_tickers = df_period.columns.get_level_values(0).unique()
                df_period = df_period.reindex(columns=all_tickers, fill_value=np.nan)

                df_period.columns.name = None  # Flatten MultiIndex properly
            except KeyError:
                logger.warning(
                    "Adj Close column not found. Returning original DataFrame."
                )

        # Align stocks with different start dates properly
        df_period = df_period.reindex(
            index=all_dates, columns=valid_symbols, fill_value=np.nan
        )
        df_period.columns.name = None  # Remove column name (Ticker)
        df_period.index.name = "Date"  # Set index name for clarity

        # Prevent removal of stocks due to shorter histories
        if config.test_mode:
            df_period.to_csv("full_df.csv")
            visible_length = int(len(df_period) * config.test_data_visible_pct)
            df_period = df_period.head(visible_length)
            logger.info(
                f"Test mode active: saved full_df.csv and limited data to {visible_length} records."
            )

        run_optimization_and_save(
            df=df_period,
            config=config,
            start_date=start,
            end_date=end,
            symbols=valid_symbols,
            stack=stack,
            years=period,
        )

    logger.info("Post-processing optimization results...")

    if not stack:
        logger.warning("No optimization results found.")
        return {}

    # Step 7: Post-processing of optimization results
    normalized_avg_weights = perform_post_processing(stack)
    if not normalized_avg_weights:
        return {}

    # Step 8: Prepare output data
    valid_models = [
        model for models in config.models.values() if models for model in models
    ]
    combined_models = ", ".join(sorted(set(valid_models)))
    combined_input_files = ", ".join(config.input_files)

    # Debugging: Log the normalized_avg_weights keys and dfs["data"] columns
    logger.debug(
        f"Normalized Average Weights Keys ({len(normalized_avg_weights)}): {list(normalized_avg_weights.keys())}"
    )
    logger.debug(
        f"dfs['data'] Columns ({len(dfs['data'].columns)}): {list(dfs['data'].columns)}"
    )

    # Ensure we only drop stocks **below the min weight threshold**, not just because of missing data
    dfs["data"] = dfs["data"].filter(items=normalized_avg_weights.keys())

    logger.debug(
        f"Filtered symbols after trimming allocations below minimum weight {config.min_weight}: {dfs['data'].columns}"
    )

    # Prevent empty DataFrame after filtering
    if dfs["data"].empty:
        logger.error("No valid symbols remain in the DataFrame after alignment.")

    # Proceed to output
    daily_returns, cumulative_returns = output(
        data=dfs["data"],
        allocation_weights=normalized_avg_weights,
        inputs=combined_input_files,
        start_date=dfs["start"],
        end_date=dfs["end"],
        optimization_model=combined_models,
        time_period=sorted_time_periods[0],
        config=config,
    )

    sorted_symbols = sorted(
        normalized_avg_weights.keys(),
        key=lambda symbol: normalized_avg_weights.get(symbol, 0),
        reverse=True,
    )

    # Step 9: Optional plotting
    if run_local:
        plot_graphs(
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            config=config,
            symbols=sorted_symbols,
        )

    # Step 10: Cleanup
    cleanup_cache("cache")
    logger.info("Pipeline execution completed successfully.")

    # Step 11: Compile and return results
    boxplot_stats = generate_boxplot_data(daily_returns)

    return {
        "start_date": str(dfs["start"]),
        "end_date": str(dfs["end"]),
        "models": combined_models,
        "symbols": sorted_symbols,
        "normalized_avg": normalized_avg_weights,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
        "boxplot_stats": boxplot_stats,  # (not extractable from plotly boxplot)
    }
