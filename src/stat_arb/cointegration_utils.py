import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from stat_arb.cointegrated_basket import johansen_cointegration
from utils.caching_utils import compute_ticker_hash


def get_cointegration_vector(
    returns_df: pd.DataFrame, cache_path: Path, reoptimize: bool = False
) -> np.ndarray:
    """
    Retrieve or compute the cointegration vector based on the returns_df.
    It uses a hashed filename based on the asset tickers to cache results.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        cache_path (Path): Path to the cache directory.
        reoptimize (bool): Whether to force re-computation.

    Returns:
        np.ndarray: The cointegration vector.
    """
    # Create cache path if it doesn't exist
    cache_path.mkdir(parents=True, exist_ok=True)

    # Compute a hash based on tickers (and possibly date range or other parameters)
    tickers = returns_df.columns.tolist()
    ticker_hash = compute_ticker_hash(tickers)  # your own function
    cache_filename = cache_path / f"cointegration_vector_{ticker_hash}.pkl"

    if not reoptimize and cache_filename.exists():
        with open(cache_filename, "rb") as f:
            cointegration_vector = pickle.load(f)
        return cointegration_vector
    else:
        # Convert returns to log prices
        log_prices = returns_df.cumsum()
        cointegration_vector = johansen_cointegration(log_prices)
        with open(cache_filename, "wb") as f:
            pickle.dump(cointegration_vector, f)
        return cointegration_vector
