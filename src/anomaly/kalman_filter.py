from typing import Dict, Tuple
import pandas as pd
import numpy as np
from filterpy import kalman
from pykalman import KalmanFilter


def apply_kalman_filter(returns_series: pd.Series, threshold: float):
    """
    Applies a Kalman filter with auto-tuning to a series of returns and detects anomalies.

    Args:
        returns_series (pd.Series): Series of returns for a single stock.
        threshold (float): Multiplier for standard deviation.

    Returns:
        Tuple[pd.Series, np.ndarray]: Anomaly flags and Kalman estimates.
    """
    # Auto-tune the Kalman filter using Expectation-Maximization (EM)
    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],  # Same as your F matrix
        observation_matrices=[[1, 0]],  # Same as your H matrix
        transition_covariance=np.array(
            [[1e-5, 0], [0, 1e-4]]
        ),  # Q matrix (process noise)
        observation_covariance=np.array([[1e-2]]),  # R matrix (measurement noise)
        initial_state_mean=[returns_series.iloc[0], 0],
        n_dim_obs=1,
        n_dim_state=2,
    )

    # Fit using past data (Expectation-Maximization auto-learns parameters)
    kf = kf.em(returns_series.values, n_iter=10)

    # Apply Kalman filtering
    filtered_state_means, _ = kf.filter(returns_series.values)

    # Compute residuals and standard deviation
    residuals = returns_series.values - filtered_state_means[:, 0]
    rolling_std_dev = pd.Series(residuals).rolling(window=30, min_periods=10).std()

    # Identify anomalies
    anomaly_flags = np.abs(residuals) > (
        threshold * rolling_std_dev.fillna(rolling_std_dev.mean())
    )

    return (
        pd.Series(anomaly_flags, index=returns_series.index),
        filtered_state_means[:, 0],
    )


# manual tuning with filterpy
# def apply_kalman_filter(
#     returns_series: pd.Series, threshold: float
# ) -> Tuple[pd.Series, np.ndarray]:
#     """
#     Applies the Kalman filter to a series of returns and returns anomaly flags and estimates.

#     Args:
#         returns_series (pd.Series): Series of returns for a single stock.
#         threshold (float): Threshold multiplier for standard deviation.

#     Returns:
#         Tuple[pd.Series, np.ndarray]: Boolean series indicating anomalies and Kalman estimates.
#     """
#     # Initialize Kalman Filter
#     kf = kalman(dim_x=2, dim_z=1)
#     kf.x = np.array(
#         [returns_series.mean(), 0.0]
#     )  # initial state (location and velocity)
#     kf.F = np.array([[1, 1], [0, 1]])  # state transition matrix
#     kf.H = np.array([[1, 0]])  # Measurement function
#     kf.P *= 1000.0  # covariance matrix
#     kf.R = 0.01  # measurement noise
#     kf.Q = np.array([[1, 0], [0, 1]]) * 0.001  # process noise

#     estimates = []
#     for z in returns_series:
#         kf.predict()
#         kf.update(z)
#         estimates.append(kf.x[0])

#     estimates = np.array(estimates)
#     residuals = returns_series.values - estimates
#     std_dev = residuals.std()
#     anomaly_flags = np.abs(residuals) > (threshold * std_dev)

#     return pd.Series(anomaly_flags, index=returns_series.index), estimates
