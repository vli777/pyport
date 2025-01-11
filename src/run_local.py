# file: cli_main.py

from core import run_pipeline
from utils import logger

def main():    
    config_file = "config.yaml"

    results = run_pipeline(config_file, run_local=True)

    logger.info("Pipeline complete. Final results summary:")
    print(results)

if __name__ == "__main__":
    main()
