from typing import Tuple
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import seaborn as sns


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


def remove_anomalous_stocks(
    returns_df: pd.DataFrame, threshold: int = 7.0, plot: bool = False
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Removes stocks with anomalous returns based on the Kalman filter.
    Optionally plots anomalies for all flagged stocks in a paginated 6x6 grid.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns.
        threshold (float): Number of standard deviations to flag anomalies.
        plot (bool): If True, anomalies will be plotted in a paginated grid.

    Returns:
        pd.DataFrame: Filtered DataFrame with anomalous stocks removed.
    """
    anomalous_cols = []

    # Dictionaries to store data for plotting if needed
    returns_data = {}
    anomaly_flags_data = {}

    for stock in returns_df.columns:
        returns_series = returns_df[stock].dropna()
        if returns_series.empty:
            print(f"Warning: No data for stock {stock}. Skipping.")
            continue

        anomaly_flags = apply_kalman_filter(returns_series, threshold=threshold)

        # If anomalies found for the stock
        if anomaly_flags.any():
            anomalous_cols.append(stock)
            # Store data for plotting if plot is True
            if plot:
                returns_data[stock] = returns_series
                anomaly_flags_data[stock] = anomaly_flags

    print(
        f"Removing {len(anomalous_cols)} stocks with Kalman anomalies: {anomalous_cols}"
    )

    # If plotting is requested and there are stocks with anomalies, plot them
    if plot and returns_data:
        # Use the list of anomalous stocks for plotting
        plot_anomalies(
            stocks=anomalous_cols,
            returns_data=returns_data,
            anomaly_flags_data=anomaly_flags_data,
            stocks_per_page=36,
        )

    # Return the DataFrame with anomalous stocks removed
    filtered_df = returns_df.drop(columns=anomalous_cols)
    return filtered_df, anomalous_cols


def plot_anomalies(stocks, returns_data, anomaly_flags_data, stocks_per_page=36):
    """
    Plots multiple stocks' return series in paginated 6x6 grids and highlights anomalies using Seaborn.

    Args:
        stocks (list): List of stock names.
        returns_data (dict): Dictionary of daily returns for each stock, keyed by stock name.
        anomaly_flags_data (dict): Dictionary of anomaly flags (np.array) for each stock, keyed by stock name.
        stocks_per_page (int): Maximum number of stocks to display per page (default is 36).
    """
    grid_rows, grid_cols = 6, 6  # Fixed grid dimensions
    total_plots = grid_rows * grid_cols
    num_pages = (
        len(stocks) + stocks_per_page - 1
    ) // stocks_per_page  # Calculate pages

    for page in range(num_pages):
        # Determine the range of stocks for this page
        start_idx = page * stocks_per_page
        end_idx = min(start_idx + stocks_per_page, len(stocks))
        stocks_to_plot = stocks[start_idx:end_idx]

        # Create a figure for this page
        fig, axes = plt.subplots(
            grid_rows, grid_cols, figsize=(18, 18), sharex=False, sharey=False
        )
        axes = axes.flatten()  # Flatten for easier indexing

        for i, stock in enumerate(stocks_to_plot):
            ax = axes[i]

            # Extract data for this stock using dictionary keys
            if stock in returns_data:
                returns_series = returns_data[stock]
            else:
                print(f"Warning: {stock} not found in returns_data.")
                continue  # Skip this stock if data is missing

            anomaly_flags = anomaly_flags_data[stock]

            # Convert anomaly_flags to a Pandas Series aligned with returns_series
            anomaly_flags_series = pd.Series(anomaly_flags, index=returns_series.index)

            # Initialize Kalman filter and perform smoothing
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            values = returns_series.values.reshape(-1, 1)
            kf = kf.em(values, n_iter=10)
            smoothed_state_means, smoothed_state_covariances = kf.smooth(values)

            # Calculate 95% confidence intervals
            mean = smoothed_state_means.squeeze()
            std_dev = np.sqrt(smoothed_state_covariances.squeeze())
            lower_bounds = mean - 1.96 * std_dev
            upper_bounds = mean + 1.96 * std_dev

            # Create a DataFrame for easier plotting
            plot_df = pd.DataFrame(
                {
                    "Date": returns_series.index,
                    "Returns": returns_series.values,
                    "Anomaly": anomaly_flags_series,
                }
            )

            # Plot the observed returns
            sns.lineplot(
                ax=ax,
                data=plot_df,
                x="Date",
                y="Returns",
                color="blue",
                label="Observed Returns",
            )

            # Overlay the Kalman smoothed mean
            ax.plot(plot_df["Date"], mean, color="green", label="Kalman Smoothed Mean")

            # Fill between the confidence intervals
            ax.fill_between(
                plot_df["Date"],
                lower_bounds,
                upper_bounds,
                color="gray",
                alpha=0.3,
                label="95% Confidence Interval",
            )

            # Highlight anomalies
            sns.scatterplot(
                ax=ax,
                data=plot_df[plot_df["Anomaly"]],
                x="Date",
                y="Returns",
                color="red",
                s=20,
                label="Anomalies",
            )

            # Simplify x-axis: Show only start and end dates
            start_date = returns_series.index.min()
            end_date = returns_series.index.max()
            ax.set_xticks([start_date, end_date])
            ax.set_xticklabels(
                [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
            )

            # Customize each subplot
            ax.set_title(stock, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend(fontsize=8)
            ax.grid(True)

        # Remove unused subplots
        for j in range(len(stocks_to_plot), total_plots):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.suptitle(f"Page {page + 1} of {num_pages}", fontsize=16)
        plt.show()
