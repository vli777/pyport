# src/utils/caching_utils.py

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

import pytz

from utils.date_utils import find_valid_trading_date
from .logger import logger


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


def make_cache_key(method, years, symbols, config_hash):
    # config_hash can be an MD5 of the config dictionary or something
    # e.g. str(sorted(config.items()))
    sorted_symbols = "_".join(sorted(symbols))
    return f"{method}_{years}_{sorted_symbols}_{config_hash}.json"


def load_model_results_from_cache(cache_key):
    """
    Load cached results if they are valid for the current trading day.

    :param cache_key: (str) Unique identifier for the cache file.
    :return: (dict) Cached results if valid, otherwise None.
    """
    cache_file = Path("cache") / cache_key
    est = pytz.timezone("US/Eastern")
    now_est = datetime.now(est)

    if cache_file.exists():
        # Check cache timestamp
        cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=est)

        # Get the most recent trading day
        most_recent_trading_day = find_valid_trading_date(
            now_est.date(), tz=est, direction="backward"
        )

        # Validate if cache is from the current trading day
        if cache_mtime.date() == most_recent_trading_day:
            with open(cache_file, "r") as f:
                return json.load(f)  # or pickle.load

    # Cache is invalid or doesn't exist
    return None


def save_model_results_to_cache(cache_key, weights_dict):
    cache_file = Path("cache") / cache_key
    with open(cache_file, "w") as f:
        json.dump(weights_dict, f, indent=4)
