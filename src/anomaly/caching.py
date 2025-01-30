import os
import pickle
from typing import Dict


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
