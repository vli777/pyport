# src/utils/caching_utils.py

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from typing import Dict, Optional, Any
import pandas as pd
import pytz

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


def make_cache_key(method, years, symbols):
    symbols_hash = hashlib.md5("_".join(sorted(symbols)).encode()).hexdigest()
    cache_file = f"{method}_{years}_{symbols_hash}.json"
    return cache_file


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

        # Allow 72 hrs cache validity
        time_diff = now_est - cache_mtime
        if time_diff.total_seconds() < 72 * 3600:
            # Check if file is not empty
            if cache_file.stat().st_size == 0:
                return None
            # Attempt to load JSON, return None on failure
            with open(cache_file, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return None

    # Cache is invalid, doesn't exist, or empty/invalid
    return None


def save_model_results_to_cache(cache_key, weights_dict):
    cache_file = Path("cache") / cache_key
    with open(cache_file, "w") as f:
        json.dump(weights_dict, f, indent=4)


def save_thresholds_to_pickle(
    thresholds: Dict[str, float], filename: str = "optimized_thresholds.pkl"
):
    """
    Saves the thresholds dictionary to a Pickle file.

    Args:
        thresholds (Dict[str, float]): Dictionary of optimized thresholds per ticker.
        filename (str): Filename for the Pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump(thresholds, f)
    print(f"Saved optimized thresholds to {filename}")


def load_thresholds_from_pickle(
    filename: str = "optimized_thresholds.pkl",
) -> Dict[str, float]:
    """
    Loads the thresholds dictionary from a Pickle file.

    Args:
        filename (str): Filename of the Pickle file.

    Returns:
        Dict[str, float]: Dictionary of optimized thresholds per ticker.
    """
    if not os.path.exists(filename):
        print(f"No cache file found at {filename}. Starting fresh optimization.")
        return {}
    with open(filename, "rb") as f:
        thresholds = pickle.load(f)
    print(f"Loaded optimized thresholds from {filename}")
    return thresholds
