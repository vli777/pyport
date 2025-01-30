from typing import Dict, Tuple
import pandas as pd
import numpy as np
from pykalman import KalmanFilter


def apply_kalman_filter(returns_series: pd.Series, threshold: float) -> Tuple[pd.Series, np.ndarray]:
    """
    Applies the Kalman filter to a series of returns and returns anomaly flags and estimates.

    Args:
        returns_series (pd.Series): Series of returns for a single stock.
        threshold (float): Threshold multiplier for standard deviation.

    Returns:
        Tuple[pd.Series, np.ndarray]: Boolean series indicating anomalies and Kalman estimates.
    """
    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([returns_series.mean(), 0.])  # initial state (location and velocity)
    kf.F = np.array([[1, 1],
                     [0, 1]])  # state transition matrix
    kf.H = np.array([[1, 0]])  # Measurement function
    kf.P *= 1000.  # covariance matrix
    kf.R = 0.01  # measurement noise
    kf.Q = np.array([[1, 0],
                     [0, 1]]) * 0.001  # process noise

    estimates = []
    for z in returns_series:
        kf.predict()
        kf.update(z)
        estimates.append(kf.x[0])

    estimates = np.array(estimates)
    residuals = returns_series.values - estimates
    std_dev = residuals.std()
    anomaly_flags = np.abs(residuals) > (threshold * std_dev)

    return pd.Series(anomaly_flags, index=returns_series.index), estimates