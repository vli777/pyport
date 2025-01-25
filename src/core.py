# file: src/core.py

from pathlib import Path
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from config import Config
from plotly_graphs import plot_graphs
from portfolio_optimization import run_optimization_and_save
from process_symbols import process_symbols
from result_output import output
from anomaly_detection import remove_anomalous_stocks

from correlation.decorrelation import filter_correlated_groups
from apply_recommendation_filter import filter_with_reversion
from correlation.optimize_correlation import optimize_correlation_threshold
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
            adj_close = df.xs("Adj Close", level=1, axis=1)
            returns = adj_close.pct_change().dropna()
            logger.debug("Calculated daily returns.")
            return returns
        except KeyError:
            logger.error("Adjusted close prices not found in the DataFrame.")
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
                threshold=config.anomaly_detection_deviation_threshold,
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

        # Step 1: Apply mean reversion filter (if enabled)
        if config.use_reversion_filter:
            filtered_symbols = filter_with_reversion(returns_df, plot=config.plot_reversion_threshold)
        else:
            filtered_symbols = original_symbols

        # Step 2: Validate filtered symbols against returns_df columns
        valid_symbols = [
            symbol for symbol in filtered_symbols if symbol in original_symbols
        ]
        if len(valid_symbols) < len(filtered_symbols):
            missing_symbols = set(filtered_symbols) - set(valid_symbols)
            logger.warning(f"Filtered symbols not in returns_df: {missing_symbols}")

        # Step 3: Apply correlation filter (if enabled)
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
                    plot=config.plot_clustering
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

        # Step 4: Ensure a non-empty list of symbols is returned
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
    returns_df, removed_anomalous = preprocess_data(df_all)
    # Log removed anomalous symbols
    if removed_anomalous:
        logger.info(f"Anomalous symbols removed: {removed_anomalous}")

    # Step 5: Filter symbols based on mean reversion only if required
    logger.info("Filtering symbols...")
    valid_symbols = filter_symbols(returns_df, config)
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

        # Slice price data for the period
        df_period = df_all.loc[start:end].copy()

        # Flatten df_period for optimization
        if isinstance(
            df_period.columns, pd.MultiIndex
        ) and "Adj Close" in df_period.columns.get_level_values(1):
            df_period = df_period.xs("Adj Close", level=1, axis=1)

        # Ensure the DataFrame is flat with symbols as columns and dates as the index
        df_period.columns.name = None  # Remove column name (Ticker)
        df_period.index.name = "Date"  # Set index name for clarity

        # Filter df_period to include only valid_symbols
        df_period = df_period[valid_symbols]

        # Update dfs with the new start and end dates
        dfs["start"] = min(dfs["start"], start)
        dfs["end"] = max(dfs["end"], end)

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

    # Align dfs["data"] with normalized_avg_weights dict
    dfs["data"] = dfs["data"].filter(items=normalized_avg_weights.keys())
    logger.debug(
        f"Filtered symbols after trimming allocations below minimum weight {config.min_weight}: {dfs['data'].columns}"
    )

    # Check for empty DataFrame after filtering
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
        minimum_weight=config.min_weight,
        max_size=config.portfolio_max_size,
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
    return {
        "start_date": str(dfs["start"]),
        "end_date": str(dfs["end"]),
        "models": combined_models,
        "symbols": sorted_symbols,
        "normalized_avg": normalized_avg_weights,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
    }
