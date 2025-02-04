import sys
from typing import List, Optional
from config import Config
from core import run_pipeline
from plotly_graphs import plot_graphs
from utils import logger


def iterative_pipeline_runner(
    config: Config,
    initial_symbols: Optional[List[str]] = None,
    max_epochs: Optional[int] = 1,
    min_weight: Optional[float] = None,
    portfolio_max_size: Optional[int] = None,
    top_n_candidates: Optional[int] = None,
    run_local: bool = False,
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
        )

        # Exclude the simulated portfolio symbol from the next epoch
        valid_symbols = [symbol for symbol in result["symbols"] if symbol != "SIM_PORT"]

        print(f"\nTop symbols from epoch {epoch + 1}: {valid_symbols}")

        # Log the number of symbols
        logger.info(
            f"Epoch {epoch + 1}: Selected {len(valid_symbols)} symbols for the next iteration."
        )

        # Check for convergence
        if set(valid_symbols) == previous_top_symbols:
            print("Convergence reached. Stopping epochs.")

            # Trim to portfolio_max_size if needed
            if len(valid_symbols) > config.portfolio_max_size:
                print(
                    f"Trimming symbols from {len(valid_symbols)} to {config.portfolio_max_size} for final portfolio."
                )
                final_result = run_pipeline(
                    config=config,
                    symbols_override=valid_symbols[: config.portfolio_max_size],
                )
            else:
                final_result = result  # No trimming needed

            break

        if len(valid_symbols) <= config.portfolio_max_size:
            print(
                f"Stopping epochs as the number of portfolio holdings ({len(valid_symbols)}) is <= the configured portfolio max size of {config.portfolio_max_size}."
            )
            # Run the final pipeline trimmed to user-defined portfolio max-size
            final_result = run_pipeline(
                config=config,
                symbols_override=valid_symbols[: config.portfolio_max_size],
            )
            break

        # Update symbols for the next epoch
        previous_top_symbols = set(valid_symbols)
        symbols = valid_symbols
        final_result = result

    # Plot only the last result if `run_local` is enabled
    if run_local:
        plot_graphs(
            daily_returns=final_result["daily_returns"],
            cumulative_returns=final_result["cumulative_returns"],
            config=config,
            symbols=final_result["symbols"],
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
