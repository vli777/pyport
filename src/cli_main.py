# file: cli_main.py

from core import run_pipeline
from config import Config
from utils import logger

def main():    
    config_file = "config.yaml"
    config =  Config.from_yaml(config_file)
    results = run_pipeline(config, symbols_override=None, run_local=True)
    logger.info("Pipeline complete. Final results summary:")
    results_summary = {
        "start_date": results["start_date"],
        "end_date": results["end_date"],
        "models": results["models"],
        "symbols": results["symbols"],
    }
    logger.info(results_summary)

    # Format normalized_avg for 2-column output
    logger.info("Normalized Average Weights (Symbol | Weight):")
    normalized_avg = results["normalized_avg"]
    sorted_weights = sorted(normalized_avg.items(), key=lambda kv: kv[1], reverse=True)
    for symbol, weight in sorted_weights:
        print(f"{symbol}\t{weight:.3f}")

if __name__ == "__main__":
    main()
