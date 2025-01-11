# src/utils/caching_utils.py

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from .logger import logger


def save_model_results(
    model_name: str,
    time_period: str,
    input_filename: str,
    symbols: list,
    scaled: Dict[str, float],
    cache_dir: Optional[Path] = None,
) -> None:
    """
    Saves the model results to cache in JSON format.

    Args:
        model_name (str): Name of the ML model.
        time_period (str): Time period identifier.
        input_filename (str): Name of the input file.
        symbols (List[str]): List of asset symbols.
        scaled (Dict[str, float]): Dictionary of asset weights.
        cache_dir (Optional[Path], optional): Directory to store cache files. Defaults to 'cache'.
    """
    cache_dir = cache_dir or Path.cwd() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_file = cache_dir / f"{input_filename}-{model_name}-{time_period}.json"

    logger.info(f"Saving results to model cache: {output_file}")

    # Initialize all symbols with 0
    results = {symbol: 0.0 for symbol in symbols}

    # Update with scaled weights
    results.update(scaled)

    try:
        with output_file.open("w") as jsonfile:
            json.dump(results, jsonfile, indent=4)
        logger.debug(f"Model results saved successfully to {output_file}.")
    except Exception as e:
        logger.error(f"Failed to save model results to {output_file}: {e}")
        raise


def load_model_results_from_cache(
    model_name: str,
    time_period: str,
    input_filename: str,
    symbols: list,
    cache_dir: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """
    Loads the model results from cache if available and valid.

    Args:
        model_name (str): Name of the ML model.
        time_period (str): Time period identifier.
        input_filename (str): Name of the input file.
        symbols (List[str]): List of asset symbols.
        cache_dir (Optional[Path], optional): Directory to read cache files from. Defaults to 'cache'.

    Returns:
        Optional[Dict[str, float]]: The cached results if valid, else None.
    """
    cache_dir = cache_dir or Path.cwd() / "cache"
    cache_file = cache_dir / f"{input_filename}-{model_name}-{time_period}.json"

    if cache_file.exists():
        logger.info(f"Loading results from cache: {cache_file}")
        try:
            with cache_file.open("r") as jsonfile:
                results = json.load(jsonfile)

            # Validate symbols
            cached_symbols = set(results.keys())
            provided_symbols = set(symbols)

            if cached_symbols == provided_symbols:
                logger.debug("Cached symbols match the provided symbols.")
                return results
            else:
                missing_in_cache = provided_symbols - cached_symbols
                missing_in_provided = cached_symbols - provided_symbols

                if missing_in_cache:
                    logger.warning(
                        f"Symbols {missing_in_cache} are missing in cached results."
                    )
                if missing_in_provided:
                    logger.warning(
                        f"Symbols {missing_in_provided} are present in cached results but not provided."
                    )

                logger.info("Symbol mismatch detected. Recalculating model results.")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {cache_file}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"An error occurred while loading cache from {cache_file}: {e}"
            )
            return None
    else:
        logger.info(f"No cache found for {model_name} with time period {time_period}.")
        return None


def cleanup_cache(cache_dir: Optional[str] = None, max_age_hours: int = 24) -> None:
    """
    Removes files in the cache directory that are older than max_age_hours.

    Args:
        cache_dir (Optional[str], optional): Path to the cache directory. Defaults to 'cache'.
        max_age_hours (int, optional): Maximum age of cache files in hours. Defaults to 24.
    """
    cache_path = Path(cache_dir) if cache_dir else Path.cwd() / "cache"

    if not cache_path.is_dir():
        logger.warning(f"Cache directory {cache_path} does not exist.")
        return

    now = datetime.now()
    max_age = timedelta(hours=max_age_hours)

    logger.info(
        f"Cleaning up cache: Removing files older than {max_age_hours} hours from {cache_path}"
    )

    for filepath in cache_path.iterdir():
        if filepath.is_file():
            try:
                modification_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                file_age = now - modification_time
                if file_age > max_age:
                    filepath.unlink()
                    logger.info(f"Removed cache file: {filepath} (Age: {file_age})")
            except Exception as e:
                logger.error(f"Failed to remove cache file {filepath}: {e}")
