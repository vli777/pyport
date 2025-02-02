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
from reversion.multiscale_reversion import apply_mean_reversion_multiscale
from reversion.optimize_timescale_weights import find_optimal_weights
from boxplot import generate_boxplot_data
from correlation.tsne_dbscan import filter_correlated_groups_dbscan
from stat_arb.stat_arb_adjust import (
    adjust_allocation_with_stat_arb,
    calculate_composite_signal,
)
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

    def preprocess_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """
        Preprocess the input DataFrame to calculate returns and optionally remove anomalous stocks.

        Args:
            df (pd.DataFrame): Multiindex DataFrame with adjusted close prices.
            config: Configuration object with settings like anomaly detection threshold and plotting options.

        Returns:
            pd.DataFrame: Processed DataFrame with daily returns, with optional anomaly filtering and decorrelation applied.
        """
        returns_df = calculate_returns(df)

        if config.use_anomaly_filter:
            logger.debug("Applying anomaly filter.")
            valid_symbols = remove_anomalous_stocks(
                returns_df=returns_df,
                cache_filename="optuna_cache/anomaly_thresholds.pkl",
                reoptimize=False,
                plot=config.plot_anomalies,
                contamination=0.02,
                max_anomaly_fraction=0.02,
            )
            filtered_returns_df = returns_df[valid_symbols]
        else:
            filtered_returns_df = returns_df

        # Apply decorrelation filter if enabled
        if config.use_decorrelation:
            logger.info("Filtering correlated assets...")
            valid_symbols = filter_correlated_assets(filtered_returns_df, config)

            valid_symbols = [
                symbol
                for symbol in valid_symbols
                if symbol in filtered_returns_df.columns
            ]

            return filtered_returns_df[valid_symbols]

        return filtered_returns_df

    def filter_correlated_assets(returns_df: pd.DataFrame, config: Config) -> List[str]:
        """
        Apply mean reversion and decorrelation filters to return valid asset symbols.
        Falls back to the original returns_df columns if filtering results in an empty list.
        """
        original_symbols = list(returns_df.columns)  # Preserve original symbols

        try:
            decorrelated_tickers = filter_correlated_groups_dbscan(
                returns_df=returns_df,
                risk_free_rate=config.risk_free_rate,
                eps=0.2,
                min_samples=2,
                top_n_per_cluster=config.top_n_candidates,
                plot=config.plot_clustering,
                cache_dir="optuna_cache",
                reoptimize=False,
            )

            valid_symbols = [
                symbol for symbol in valid_symbols if symbol in decorrelated_tickers
            ]

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

    def stat_arb_adjust(
        baseline_allocation: pd.Series, returns_df: pd.DataFrame, config: Config
    ) -> pd.Series:
        """
        Apply z-score based allocation weight adjustments
        Args:
            baseline_allocation (pd.Series) Original weight allocation after optimization
            returns_df (pd.DataFrame): The returns dataframe of all selected stocks in the portfolio
            config (Config): yaml config loader object

        Returns:
            final_allocation (pd.Series)
        """
        # Apply z-score based allocation weight adjustments

        # Step 1. Generate reversion signals (using your multiscale implementation)
        reversion_signals = apply_mean_reversion_multiscale(
            returns_df, n_trials=50, n_jobs=-1, plot=config.plot_reversion_threshold
        )
        print("Reversion Signals Generated.")

        # Step 2. Optimize weights for the reversion signals (for example, the daily/weekly weighting)
        optimal_weights = find_optimal_weights(
            reversion_signals, returns_df, n_trials=50
        )
        print(f"Optimal Weights: {optimal_weights}")

        # Step 3. Compute the composite stat arb signal from the reversion signals.
        # (This gives a continuous signal for each ticker.)
        composite_signals = calculate_composite_signal(
            reversion_signals,
            weight_daily=optimal_weights.get("weight_daily", 0.5),
            weight_weekly=1.0 - optimal_weights.get("weight_daily", 0.5),
        )
        print(f"Composite Signals: {composite_signals}")

        final_allocation = adjust_allocation_with_stat_arb(
            baseline_allocation=baseline_allocation,
            composite_signals=composite_signals,
            alpha=0.2,  # e.g. 0.2 or another value determined by testing
            allow_short=False,  # config.allow_short,
        )

        return final_allocation

    def perform_post_processing(
        stack_weights: Dict[str, Any], returns_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Perform post-processing on the stack data to calculate normalized weights.

        Args:
            stack_weights (Dict[str, Any]): The stack weights data containing optimization results.
            returns_df (pd.DataFrame): Returns of the assets in the portfolio

        Returns:
            Dict[str, Any]: Normalized weights as a dictionary.
        """
        # Convert pd.Series to dictionaries if necessary
        processed_stack = {
            key: (value.to_dict() if isinstance(value, pd.Series) else value)
            for key, value in stack_weights.items()
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

        if config.use_mean_reversion:
            normalized_weights = stat_arb_adjust(
                baseline_allocation=normalized_weights,
                returns_df=returns_df,
                config=config,
            )

        # Ensure output is a dictionary
        if isinstance(normalized_weights, pd.Series):
            normalized_weights = normalized_weights.to_dict()

        return normalized_weights

    # Load and validate symbols
    try:
        all_symbols = load_symbols()
        if not all_symbols:
            logger.warning("No symbols found. Aborting pipeline.")
            return {}
    except ValueError as e:
        logger.error(f"Symbol override validation failed: {e}")
        return {}

    # Initialize structures
    stack: Dict[str, Any] = {}
    dfs: Dict[str, Any] = {}
    active_models = [k for k, v in config.models.items() if v]
    sorted_time_periods = sorted(active_models, reverse=True)

    # Determine date range based on the longest period
    longest_period = sorted_time_periods[0]
    start_long, end_long = calculate_start_end_dates(longest_period)
    dfs["start"] = start_long
    dfs["end"] = end_long

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df_all = load_data(all_symbols, start_long, end_long)

    # Ensure we use the **largest valid date range** for returns_df
    all_dates = df_all.index  # Keep full range before filtering
    returns_df = preprocess_data(df_all, config)  # apply all pre-optimization filters

    valid_symbols = list(returns_df.columns)
    if not valid_symbols:
        logger.warning("No valid symbols remain after filtering. Aborting pipeline.")
        return None  # Or raise an exception if this is an unrecoverable state

    logger.info(f"Symbols selected for optimization: {valid_symbols}")

    try:
        if valid_symbols:
            dfs["data"] = df_all.xs("Adj Close", level=1, axis=1)[valid_symbols]
            logger.debug(f"dfs['data'] shape: {dfs['data'].shape}")
        else:
            logger.warning("No valid symbols available for slicing df_all.")
            dfs["data"] = pd.DataFrame()  # Or handle gracefully
    except KeyError as e:
        logger.error(f"Error slicing df_all with filtered_decorrelated: {e}")
        raise

    # Iterate through each time period and perform optimization
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

    # Post-processing of optimization results
    normalized_avg_weights = perform_post_processing(stack, returns_df)
    if not normalized_avg_weights:
        return {}

    # Prepare output data
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

    # Optional plotting
    if run_local:
        plot_graphs(
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            config=config,
            symbols=sorted_symbols,
        )

    # Cleanup
    cleanup_cache("cache")
    logger.info("Pipeline execution completed successfully.")

    # Compile and return results
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
