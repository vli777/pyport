from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Tuple


def apply_isolation_forest(
    returns_series: pd.Series, threshold: float = 1.5, contamination: float = "auto"
) -> Tuple[pd.Series, pd.Series]:
    """
    Applies IsolationForest to detect anomalies in a stock's return series.

    Args:
        returns_series (pd.Series): Series of returns for a single stock.
        threshold (float): Multiplier for standard deviation below the median anomaly score.
                           Higher values make anomaly detection stricter.
        contamination (float): Expected fraction of anomalies (default is 'auto').

    Returns:
        Tuple[pd.Series, pd.Series]:
            - A boolean series of anomaly flags (True = anomaly).
            - A series of anomaly scores (lower = more likely to be an anomaly).
    """
    if returns_series.isna().all():
        raise ValueError("All values in returns_series are NaN.")

    # Convert to 2D array for Isolation Forest
    X = returns_series.values.reshape(-1, 1)

    # Scale data to avoid biasing IsolationForest due to magnitude differences
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    clf = IsolationForest(
        n_estimators=100, contamination=contamination, random_state=42
    )
    clf.fit(X_scaled)

    # Get anomaly scores (higher scores = more normal, lower = more anomalous)
    scores = clf.decision_function(X_scaled)

    # Determine threshold dynamically if not provided
    if threshold is None:
        anomaly_cutoff = np.percentile(scores, 5)  # Bottom 5% as anomalies
    else:
        median_score = np.median(scores)
        std_score = np.std(scores)
        anomaly_cutoff = median_score - threshold * std_score

    # Flag anomalies
    anomaly_flags = scores < anomaly_cutoff

    return pd.Series(anomaly_flags, index=returns_series.index), pd.Series(
        scores, index=returns_series.index
    )
