import pandas as pd
import numpy as np
from pykalman import KalmanFilter


def apply_kalman_filter(returns_series, threshold=7.0, epsilon=1e-6):
    # Ensure returns_series is 1-dimensional
    if not isinstance(returns_series, pd.Series):
        raise ValueError("returns_series must be a Pandas Series.")

    # Ensure the series is valid
    if len(returns_series) < 2:
        print(f"Insufficient data for Kalman filter. Length: {len(returns_series)}")
        return pd.Series([False] * len(returns_series), index=returns_series.index)

    # Initialize the Kalman filter
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

    # Increase process noise for smoother estimates
    kf.transition_covariance = np.eye(1) * 0.1

    # Reshape the input to 2D array (T x 1)
    values = returns_series.values.reshape(-1, 1)

    # Train the Kalman filter
    kf = kf.em(values, n_iter=3)

    # Get smoothed state means
    smoothed_state_means, _ = kf.smooth(values)

    # Calculate residuals
    residuals = values - smoothed_state_means

    # Calculate the median of residuals
    median_res = np.median(residuals)

    # Compute the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(residuals - median_res))

    # Avoid division by zero or very small MAD
    mad = max(mad, epsilon)

    # Define a modified Z-score using MAD
    modified_z_scores = 0.6745 * (residuals - median_res) / mad

    # Identify anomalies based on a threshold on the modified Z-score
    anomaly_flags = np.abs(modified_z_scores) > threshold

    # Squeeze anomaly_flags to convert from 2D to 1D
    anomaly_flags = anomaly_flags.squeeze()

    return anomaly_flags
