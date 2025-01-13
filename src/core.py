# file: src/core.py

from pathlib import Path
import logging
from typing import Any, Dict, List, Optional

from config import Config
from plotly_graphs import plot_graphs
from portfolio_optimization import run_optimization_and_save
from process_symbols import process_symbols
from result_output import output
from utils.caching_utils import cleanup_cache
from utils.data_utils import process_input_files
from utils.date_utils import calculate_start_end_dates
from utils.portfolio_utils import normalize_weights, stacked_output


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
    # Set up output directories, etc.
    CWD = Path.cwd()
    PATH = CWD / config.folder
    PATH.mkdir(parents=True, exist_ok=True)

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
            Path(config.input_files_folder) / file for file in config.input_files
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

    for years in sorted_times:
        start_date, end_date = calculate_start_end_dates(years)
        if config.test_mode:
            logger.info(f"Time period: {years}, symbols: {all_symbols}")

        # Load data
        df = process_symbols(all_symbols, start_date, end_date, PATH, config.download)

        # If this is the first loop, store the big DataFrame; else update min/max dates
        if "data" not in dfs:
            dfs.update({"data": df, "start": start_date, "end": end_date})
        else:
            dfs["start"] = min(dfs["start"], start_date)
            dfs["end"] = max(dfs["end"], end_date)

        # Optional: If test_mode is on, store a CSV of the full data
        if config.test_mode:
            df.to_csv("full_df.csv")
            df = df.head(int(len(df) * config.test_data_visible_pct))

        # Run optimization
        run_optimization_and_save(
            df, config, start_date, end_date, all_symbols, stack, years
        )

    if not stack:
        logger.warning("No optimization results found.")
        return {}

    # Post-processing
    avg = stacked_output(stack)
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
        "symbols": list(daily_returns.columns),
        "normalized_avg": normalized_avg,
    }
