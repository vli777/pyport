from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

def apply_isolation_forest(returns_series: pd.Series, threshold: float):
    """
    Applies IsolationForest fit to the data and flags points as anomalous if their 
    decision function scores fall well below the norm.

    Args:
        returns_series (pd.Series): Series of returns for a single stock.
        threshold (float): Multiplier for the standard deviation of decision function scores.
                           (Typical tuned values might be in the range 1-3.)

    Returns:
        Tuple[pd.Series, np.ndarray]: A boolean series of anomaly flags (True indicates an anomaly)
                                       and the corresponding anomaly scores from the IsolationForest.
    """
    # Reshape the returns data to 2D (IsolationForest expects a 2D array)
    X = returns_series.values.reshape(-1, 1)

    # Fit IsolationForest (a modern, ensemble-based anomaly detector)
    clf = IsolationForest(contamination='auto', random_state=42)
    clf.fit(X)

    # Obtain anomaly scores.
    # Note: Higher decision_function scores imply more normal observations,
    # while lower scores indicate potential anomalies.
    scores = clf.decision_function(X)

    # Compute a cutoff: anomalies are defined as points whose score falls below
    # (median - threshold * standard_deviation). Adjust the threshold multiplier as needed.
    median_score = np.median(scores)
    std_score = np.std(scores)
    cutoff = median_score - threshold * std_score

    # Flag anomalies
    anomaly_flags = scores < cutoff

    return pd.Series(anomaly_flags, index=returns_series.index), scores
