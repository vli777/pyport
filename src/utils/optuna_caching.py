import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

EXPIRATION_DAYS = 365  # 1-year expiration period


def get_cache_dir(directory: str) -> Path:
    """
    Get or create the specified cache directory.

    Args:
        directory (str): The directory name where cache files should be stored.

    Returns:
        Path: The Path object for the cache directory.
    """
    cache_dir = Path(directory)
    cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    return cache_dir


def load_cached_thresholds(directory: str, file_name: str) -> Optional[Dict[str, Any]]:
    """
    Load cached thresholds if they exist and are not expired.

    Args:
        directory (str): Cache directory where the file is stored.
        file_name (str): Name of the cache file.

    Returns:
        Optional[Dict[str, Any]]: Cached thresholds if valid, otherwise None.
    """
    cache_dir = get_cache_dir(directory)
    cache_file = cache_dir / f"{file_name}.parquet"
    metadata_file = cache_dir / f"{file_name}_metadata.json"

    if not cache_file.exists() or not metadata_file.exists():
        return None  # No cache file found

    # Read metadata and check expiration
    with metadata_file.open("r") as f:
        metadata = json.load(f)

    last_updated = datetime.strptime(metadata["last_updated"], "%Y-%m-%d")
    if datetime.now() - last_updated > timedelta(days=EXPIRATION_DAYS):
        print(f"Cache expired for {file_name}. Running new optimization.")
        return None  # Expired cache

    # Load and return cached data
    print(f"Loading cached {file_name} thresholds...")
    return pd.read_parquet(cache_file).to_dict()


def save_cached_thresholds(directory: str, file_name: str, thresholds: Dict[str, Any]):
    """
    Save the optimized ticker thresholds to a cache file.

    Args:
        directory (str): Cache directory where the file should be stored.
        file_name (str): Name of the cache file.
        thresholds (Dict[str, Any]): Dictionary of ticker thresholds to cache.
    """
    cache_dir = get_cache_dir(directory)
    cache_file = cache_dir / f"{file_name}.parquet"
    metadata_file = cache_dir / f"{file_name}_metadata.json"

    df = pd.DataFrame.from_dict(
        thresholds, orient="index", columns=["overbought", "oversold"]
    )
    df.to_parquet(cache_file, index=True)

    # Save metadata with last updated timestamp
    metadata = {"last_updated": datetime.now().strftime("%Y-%m-%d")}
    with metadata_file.open("w") as f:
        json.dump(metadata, f)

    print(f"Saved {file_name} thresholds to cache.")
