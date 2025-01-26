import sys
from typing import List, Optional
from config import Config
from core import run_pipeline
from plotly_graphs import plot_graphs
from utils import logger


def iterative_pipeline_runner(
    config: Config,
    initial_symbols: Optional[List[str]] = None,
    max_epochs: Optional[int] = 10,
    min_weight: Optional[float] = None,
    portfolio_max_size: Optional[int] = None,
    top_n_candidates: Optional[int] = None,
    run_local: bool = True,
):
    """
    Runs the pipeline iteratively, updating config with provided arguments if valid.

    Parameters:
    ----------
    config : Config
        The configuration object for the pipeline.
    initial_symbols : list, optional
        List of initial symbols to start the pipeline.
    max_epochs : int, optional
        Maximum number of epochs.
    min_weight : float, optional
        Minimum asset allocation weight to be considered.
    portfolio_max_size : int, optional
        Number of top symbols to select for the next epoch.
    run_local : bool, optional
        Whether to run the pipeline locally.

    Returns:
    -------
    dict
        Results of the final pipeline run.
    """
    # Update config with provided arguments if they are valid
    if min_weight is not None:
        if isinstance(min_weight, float):
            config.min_weight = min_weight
        else:
            raise TypeError("min_weight must be a float")

    if portfolio_max_size is not None:
        if isinstance(portfolio_max_size, int):
            config.portfolio_max_size = portfolio_max_size
        else:
            raise TypeError("portfolio_max_size must be an integer")

    if top_n_candidates is not None:
        if isinstance(top_n_candidates, int):
            config.top_n_candidates = top_n_candidates
        else:
            raise TypeError("top_n_candidates must be an integer")

    # initial_symbols and run_local are handled separately as they may not be part of config
    symbols = initial_symbols
    previous_top_symbols = set()
    final_result = None

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}")

        # Run the pipeline
        result = run_pipeline(
            config=config,
            symbols_override=symbols,
            run_local=False,  # Only plot in the last epoch
        )

        # Exclude the simulated portfolio symbol from the next epoch
        valid_symbols = [
            symbol
            for symbol in result["symbols"][: config.portfolio_max_size]
            if symbol != "SIM_PORT"
        ]

        print(f"\nTop symbols from epoch {epoch + 1}: {valid_symbols}")

        # Check for convergence
        if set(valid_symbols) == previous_top_symbols:
            print("Convergence reached. Stopping epochs.")
            final_result = result
            break

        if len(valid_symbols) <= config.portfolio_max_size:
            print(
                f"Stopping epochs as the number of valid symbols ({len(valid_symbols)}) is <= {config.portfolio_max_size}."
            )
            final_result = result
            break

        # Update symbols for the next epoch
        previous_top_symbols = set(valid_symbols)
        symbols = valid_symbols
        final_result = result

    # After the loop, filter the cumulative_returns and daily_returns
    if final_result:
        # Extract the current valid_symbols
        current_valid_symbols = set(valid_symbols)

        # Find available symbols in daily_returns and cumulative_returns
        available_daily_symbols = set(final_result["daily_returns"].columns)
        available_cumulative_symbols = set(final_result["cumulative_returns"].columns)

        # Determine symbols present in both DataFrames
        available_symbols = current_valid_symbols & available_daily_symbols & available_cumulative_symbols

        # Identify missing symbols
        missing_symbols = current_valid_symbols - available_symbols

        if missing_symbols:
            logger.warning(f"The following symbols are missing in daily_returns or cumulative_returns and will be skipped: {missing_symbols}")

        # Update valid_symbols to include only available symbols
        valid_symbols = list(available_symbols)

        # Check if there are still valid symbols left
        if not valid_symbols:
            logger.error("No valid symbols available after filtering out missing symbols. Exiting.")
            sys.exit()

        # Filter the DataFrames
        filtered_daily_returns = final_result["daily_returns"][valid_symbols]
        filtered_cumulative_returns = final_result["cumulative_returns"][valid_symbols]
        
        # Plot only the last result if `run_local` is enabled
        if run_local:
            plot_graphs(
                daily_returns=filtered_daily_returns,
                cumulative_returns=filtered_cumulative_returns,
                config=config,
                symbols=valid_symbols,
            )

    return final_result


if __name__ == "__main__":
    config_file = "config.yaml"
    config = Config.from_yaml(config_file)

    final_result = iterative_pipeline_runner(
        config=config,
        initial_symbols=None,  # Or provide initial symbols as needed
        run_local=True,
    )

    print("\nFinal result:", final_result)
