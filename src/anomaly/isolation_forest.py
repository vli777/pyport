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

    # Compute absolute percent changes
    abs_pct_change = returns_series.pct_change().abs()

    # Rolling Z-score
    rolling_mean = returns_series.rolling(30, min_periods=5).mean()
    rolling_std = returns_series.rolling(30, min_periods=5).std()
    rolling_z_score = (returns_series - rolling_mean) / (rolling_std + 1e-8)

    # Fill NaN and clip extreme values
    abs_pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    rolling_z_score.replace([np.inf, -np.inf], np.nan, inplace=True)
    abs_pct_change.fillna(0, inplace=True)
    rolling_z_score.fillna(0, inplace=True)
    abs_pct_change = np.clip(abs_pct_change, 0, 1)  # Limit pct changes to 100%
    rolling_z_score = np.clip(rolling_z_score, -5, 5)  # Limit Z-score range

    # Stack features
    X = np.column_stack([returns_series.fillna(0), abs_pct_change, rolling_z_score])

    # Clip extreme values in X
    X = np.clip(X, -1e6, 1e6)  # Prevents float64 overflow

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    clf = IsolationForest(
        n_estimators=100, contamination=contamination, random_state=42
    )
    clf.fit(X_scaled)

    # Get anomaly scores
    scores = clf.decision_function(X_scaled)

    # Determine threshold dynamically
    median_score = np.median(scores)
    std_score = np.std(scores)
    anomaly_cutoff = median_score - max(threshold * std_score, 0.1 * std_score)
    anomaly_flags = scores < anomaly_cutoff
    anomaly_fraction = anomaly_flags.mean()

    # If volatility is high but anomaly_fraction is suspiciously low, adjust cutoff:
    if anomaly_fraction < 0.05 and np.std(returns_series) > 0.05:
        # For very volatile series, use a fixed percentile cutoff (e.g. bottom 10%).
        anomaly_cutoff = np.percentile(scores, 10)
        anomaly_flags = scores < anomaly_cutoff
        anomaly_fraction = anomaly_flags.mean()

    return pd.Series(anomaly_flags, index=returns_series.index), pd.Series(
        scores, index=returns_series.index
    )
