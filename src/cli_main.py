# file: cli_main.py

from core import run_pipeline
from config import Config
from utils import logger

def main():    
    config_file = "config.yaml"
    config =  Config.from_yaml(config_file)
    results = run_pipeline(config, symbols_override=None, run_local=True)
    logger.info("Pipeline complete. Final results summary:")
    print(results)

if __name__ == "__main__":
    main()
