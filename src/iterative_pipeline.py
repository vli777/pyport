from typing import List, Optional
from config import Config
from core import run_pipeline
from plotly_graphs import plot_graphs


def iterative_pipeline_runner(
    config,
    initial_symbols: Optional[List[str]] = None,
    max_epochs=10,
    portfolio_max_size=5,
    run_local=True,
):
    """
    Runs the pipeline iteratively, passing the top-performing symbols from each run to the next.

    Parameters:
    ----------
    config : object
        The configuration object for the pipeline.
    initial_symbols : list, optional
        List of initial symbols to start the pipeline.
    max_epochs : int, optional
        Maximum number of epochs, by default 10.
    portfolio_max_size : int, optional
        Number of top symbols to select for the next epoch, by default 5.
    run_local : bool, optional
        Whether to run the pipeline locally, by default True.

    Returns:
    -------
    dict
        Results of the final pipeline run.
    """
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
            for symbol in result["symbols"][:portfolio_max_size]
            if symbol != "SIM_PORT"  # Replace "SIM_PORT" with the actual name
        ]

        print(f"Top symbols from epoch {epoch + 1}: {valid_symbols}")

        # Check for convergence
        if set(valid_symbols) == previous_top_symbols:
            print("Convergence reached. Stopping epochs.")
            final_result = result
            break

        if len(valid_symbols) <= 10:
            print(f"Stopping epochs as the number of valid symbols ({len(valid_symbols)}) is <= 10.")
            final_result = result
            break

        # Update symbols for the next epoch
        previous_top_symbols = set(valid_symbols)
        symbols = valid_symbols
        final_result = result

    # Plot only the last result if `run_local` is enabled
    if run_local and final_result:
        plot_graphs(
            final_result["daily_returns"],
            final_result["cumulative_returns"],
            config,
            symbols=final_result["symbols"],
        )

    return final_result


if __name__ == "__main__":
    config_file = "config.yaml"
    config = Config.from_yaml(config_file)
    portfolio_max_size = config.portfolio_max_size
    initial_symbols = None

    final_result = iterative_pipeline_runner(
        config=config,
        initial_symbols=initial_symbols,
        max_epochs=10,
        portfolio_max_size=portfolio_max_size,
        run_local=True,
    )

    print("Final result:", final_result)
