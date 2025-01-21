# file: cli_main.py

from core import run_pipeline
from config import Config
from utils import logger
from utils.scraper import get_etf_holdings


def log_and_format_results(
    results, logger, header_title="Pipeline complete. Final results summary:"
):
    """
    Logs a summary of the pipeline results and prints normalized average weights.

    Args:
        results (dict): The results dictionary from the pipeline.
        logger (logging.Logger): The logger to use for output.
        header_title (str): A title to log before the summary.
    """
    logger.info(header_title)
    # Log summary
    results_summary = {
        "start_date": results["start_date"],
        "end_date": results["end_date"],
        "models": results["models"],
        "symbols": results["symbols"],
    }
    logger.info(results_summary)

    # Format and print normalized_avg
    logger.info("Normalized Average Weights (Symbol | Weight):")
    normalized_avg = results.get("normalized_avg", {})
    sorted_weights = sorted(normalized_avg.items(), key=lambda kv: kv[1], reverse=True)
    for symbol, weight in sorted_weights:
        print(f"{symbol}\t{weight:.3f}")


def main():
    config_file = "config.yaml"
    config = Config.from_yaml(config_file)

    # Run initial pipeline
    results = run_pipeline(config, symbols_override=None, run_local=True)
    log_and_format_results(results, logger)

    # If ETF expansion is enabled
    if config.expand_etfs:
        expanded_symbols = set(results["symbols"])

        for etf_symbol in results["symbols"]:
            holdings = get_etf_holdings(etf_symbol)
            if holdings:
                expanded_symbols.update(holdings)

        expanded_results = run_pipeline(
            config, symbols_override=list(expanded_symbols), run_local=True
        )
        log_and_format_results(
            expanded_results,
            logger,
            header_title="Expanded pipeline complete. Final results summary:",
        )


if __name__ == "__main__":
    main()
