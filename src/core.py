# file: app/core.py

from pathlib import Path
import logging

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

def run_pipeline(config_file: str, run_local: bool = True):
    """
    High-level orchestration of your entire pipeline:
      - Read config
      - Prepare directories
      - Process symbols, load data
      - Run optimizations
      - Plot or return results
      - Clean up cache if needed

    Args:
        config_file (str): Path to your config.yaml.
        run_local (bool): If True, print to terminal and show Plotly graphs;
                          If False, skip printing/plotting (or handle differently).

    Returns:
        Dict of final results or JSON-serializable object for API usage.
    """
    # 1) Load config
    config = Config.from_yaml(config_file)

    # 2) Setup directories
    CWD = Path.cwd()
    PATH = CWD / config.folder
    PATH.mkdir(parents=True, exist_ok=True)

    stack, dfs = {}, {}

    # 3) Determine time periods
    filtered_times = [k for k in config.models.keys() if config.models[k]]
    sorted_times = sorted(filtered_times, reverse=True)

    # 4) For each time period
    for years in sorted_times:
        start_date, end_date = calculate_start_end_dates(years)
        watchlist_files = [Path(config.input_files_folder) / file for file in config.input_files]
        symbols = process_input_files(watchlist_files)

        if config.test_mode:
            logger.info(f"Test mode: symbols - {symbols}")

        df = process_symbols(symbols, start_date, end_date, PATH, config.download)

        # Store or update dfs
        if "data" not in dfs:
            dfs["data"] = df
            dfs["start"], dfs["end"] = start_date, end_date
        else:
            dfs["start"] = min(dfs["start"], start_date)
            dfs["end"] = max(dfs["end"], end_date)

        # Optionally store a full copy for test mode
        if config.test_mode:
            df.to_csv("full_df.csv")
            # Truncate to visible pct
            df = df.head(int(len(df) * config.config["test_data_visible_pct"]))

        # Run the optimization(s) and cache results
        run_optimization_and_save(df, config, start_date, end_date, symbols, stack, years)

    # 5) Post-processing
    final_json = {}
    if stack:
        avg = stacked_output(stack)
        sorted_avg = dict(sorted(avg.items(), key=lambda item: item[1]))
        normalized_avg = normalize_weights(sorted_avg, config.config["min_weight"])

        # Gather model names
        valid_models = [v for v in config.models.values() if v]
        combined_model_names = ", ".join(sorted(list(set(sum(valid_models, [])))))
        combined_input_files_names = ", ".join(str(i) for i in sorted(config.input_files))

        # If we want to produce final output (daily & cum returns)
        daily_returns_to_plot, cumulative_returns_to_plot = output(
            data=dfs["data"],
            allocation_weights=normalized_avg,
            inputs=combined_input_files_names,
            start_date=dfs["start"],
            end_date=dfs["end"],
            optimization_model=combined_model_names,
            time_period=sorted_times[0],
            minimum_weight=config.config["min_weight"],
            max_size=config.config["portfolio_max_size"],
            config=config,
        )

        # If running locally, show or print stuff
        if run_local:
            plot_graphs(
                daily_returns_to_plot,
                cumulative_returns_to_plot,
                config,
                symbols=daily_returns_to_plot.columns.tolist(),
            )

        # Prepare a final results dict that can be returned
        final_json = {
            "start_date": str(dfs["start"]),
            "end_date": str(dfs["end"]),
            "models": combined_model_names,
            "symbols": list(daily_returns_to_plot.columns),
            "normalized_avg": normalized_avg,
        }

    # Optionally cleanup cache
    cache_dir = "cache"
    cleanup_cache(cache_dir)

    return final_json
