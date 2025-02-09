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
    plot_done = False
    reversion_plotted = False

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}")

        # Enable filters only in the first epoch
        if epoch > 0:
            config.use_anomaly_filter = False
            config.use_decorrelation = False

        # Run the pipeline
        result = run_pipeline(
            config=config,
            symbols_override=symbols,
        )

        # Exclude the simulated portfolio symbol from the next epoch
        valid_symbols = [symbol for symbol in result["symbols"] if symbol != "SIM_PORT"]

        print(f"\nTop symbols from epoch {epoch + 1}: {valid_symbols}")

        logger.info(
            f"Epoch {epoch + 1}: Selected {len(valid_symbols)} symbols for the next iteration."
        )

        # Check for convergence
        if set(valid_symbols) == previous_top_symbols:
            print("Convergence reached. Stopping epochs.")

            # Trim to portfolio_max_size if needed
            final_symbols = (
                valid_symbols[: config.portfolio_max_size]
                if len(valid_symbols) > config.portfolio_max_size
                else valid_symbols
            )
            final_result = run_pipeline(
                config=config,
                symbols_override=final_symbols,
            )
            break

        if len(valid_symbols) <= config.portfolio_max_size:
            print(
                f"Stopping epochs as the number of portfolio holdings ({len(valid_symbols)}) is <= the configured portfolio max size of {config.portfolio_max_size}."
            )

            # Use the last valid result instead of running the pipeline again
            final_result = result
            break

        # Update symbols for the next epoch
        previous_top_symbols = set(valid_symbols)
        symbols = valid_symbols
        final_result = result

    # Ensure reversion parameters are only plotted once
    if config.plot_reversion and not reversion_plotted:
        config.plot_reversion = False  # Disable further reversion plotting
        reversion_plotted = True

    # Ensure plotting is only done in the final result
    if run_local and not plot_done:
        plot_graphs(
            daily_returns=final_result["daily_returns"],
            cumulative_returns=final_result["cumulative_returns"],
            config=config,
            symbols=final_result["symbols"],
        )
        plot_done = True

    return final_result


if __name__ == "__main__":
    config_file = "config.yaml"
    config = Config.from_yaml(config_file)

    final_result = iterative_pipeline_runner(
        config=config,
        initial_symbols=None,  # Or provide initial symbols as needed
        max_epochs=1,
        run_local=True,
    )
