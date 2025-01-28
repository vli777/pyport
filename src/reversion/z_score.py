from typing import Dict, Tuple
import matplotlib.pyplot as plt
import math

import numpy as np
import pandas as pd


def calculate_z_score(returns_df, window):
    """
    Calculate the Z-Score for each ticker in a returns DataFrame based on a rolling window.

    Args:
        returns_df (pd.DataFrame): DataFrame of log returns with tickers as columns and dates as index.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: Z-Score DataFrame for all tickers.
    """
    z_scores = returns_df.apply(
        lambda x: (x - x.rolling(window=window, min_periods=1).mean()) /
                  x.rolling(window=window, min_periods=1).std(), axis=0
    )
    return z_scores



def get_zscore_thresholds_frame(
    returns_df: pd.DataFrame,
    dynamic_windows: Dict[str, Dict[str, int]],
    overbought_multipliers: Dict[str, float],
    oversold_multipliers: Dict[str, float]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Calculate dynamic Z-score thresholds for each ticker and period based on dynamic windows.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        dynamic_windows (Dict[str, Dict[str, int]]): Window sizes per period and ticker.
            Example:
            {
                'daily': {'AAPL': 20, 'GOOGL': 20, ...},
                'weekly': {'AAPL': 3, 'GOOGL': 3, ...},
            }
        overbought_multipliers (Dict[str, float]): Overbought multipliers per ticker.
        oversold_multipliers (Dict[str, float]): Oversold multipliers per ticker.

    Returns:
        Dict[str, Dict[str, Tuple[float, float]]]: Z-score thresholds per period and ticker.
            Example:
            {
                'daily': {'AAPL': (1.2, -1.5), 'GOOGL': (1.1, -1.3), ...},
                'weekly': {'AAPL': (1.3, -1.4), 'GOOGL': (1.2, -1.2), ...},
            }
    """
    thresholds = {}
    for period in ['daily', 'weekly']:
        thresholds[period] = {}
        for ticker, window in dynamic_windows.get(period, {}).items():
            rolling_mean = returns_df[ticker].rolling(window=window, min_periods=1).mean()
            rolling_std = returns_df[ticker].rolling(window=window, min_periods=1).std()
            z_scores = (returns_df[ticker] - rolling_mean) / rolling_std.replace(0, np.nan)
            z_std = z_scores.std()
            if np.isnan(z_std) or z_std == 0:
                z_std = 1.0  # Avoid division by zero

            overbought_threshold = overbought_multipliers.get(ticker, 1.0) * z_std
            oversold_threshold = oversold_multipliers.get(ticker, 1.0) * z_std
            thresholds[period][ticker] = (overbought_threshold, oversold_threshold)
    return thresholds


def get_zscore_thresholds_ticker(
    returns_df: pd.DataFrame,
    window: int,
    overbought_multiplier: float,
    oversold_multiplier: float
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate dynamic Z-score thresholds for each ticker based on a single rolling window.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        window (int): Rolling window size for Z-score calculation.
        overbought_multiplier (float): Multiplier for overbought Z-score threshold.
        oversold_multiplier (float): Multiplier for oversold Z-score threshold.

    Returns:
        Dict[str, Tuple[float, float]]: Z-score thresholds per ticker.
            Example: {'AAPL': (1.2, -1.5), 'GOOGL': (1.1, -1.3), ...}
    """
    thresholds = {}
    for ticker in returns_df.columns:
        rolling_mean = returns_df[ticker].rolling(window=window, min_periods=1).mean()
        rolling_std = returns_df[ticker].rolling(window=window, min_periods=1).std()
        z_scores = (returns_df[ticker] - rolling_mean) / rolling_std.replace(0, np.nan)
        z_std = z_scores.std()
        if np.isnan(z_std) or z_std == 0:
            z_std = 1.0  # Avoid division by zero

        overbought_threshold = overbought_multiplier * z_std
        oversold_threshold = oversold_multiplier * z_std
        thresholds[ticker] = (overbought_threshold, oversold_threshold)
    return thresholds


def plot_z_scores_grid(
    z_scores_df,
    overbought_thresholds,
    oversold_thresholds,
    grid_shape=(6, 6),
    figsize=(20, 20),
):
    """
    Plot Z-Scores for multiple tickers in a paginated grid layout.

    Args:
        z_scores_df (pd.DataFrame): DataFrame containing Z-Scores for all tickers.
                                     Columns are tickers, index are dates.
        overbought_thresholds (pd.Series): Series containing overbought thresholds for each ticker.
                                           Index should match z_scores_df columns.
        oversold_thresholds (pd.Series): Series containing oversold thresholds for each ticker.
                                         Index should match z_scores_df columns.
        grid_shape (tuple, optional): Tuple indicating the grid size (rows, cols) per page. Default is (6, 6).
        figsize (tuple, optional): Size of each figure. Default is (20, 20).

    Returns:
        None
    """
    num_tickers = len(z_scores_df.columns)
    tickers = z_scores_df.columns.tolist()
    pages = math.ceil(num_tickers / (grid_shape[0] * grid_shape[1]))

    for page in range(pages):
        start_idx = page * grid_shape[0] * grid_shape[1]
        end_idx = start_idx + grid_shape[0] * grid_shape[1]
        current_tickers = tickers[start_idx:end_idx]

        num_current = len(current_tickers)
        rows, cols = grid_shape
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()  # Flatten to 1D for easy iteration

        for i, ticker in enumerate(current_tickers):
            ax = axes[i]
            z_scores = z_scores_df[ticker]
            overbought = overbought_thresholds.get(ticker, 1.0)
            oversold = oversold_thresholds.get(ticker, -1.0)

            ax.plot(z_scores.index, z_scores, label=f"{ticker} Z-Score")
            ax.axhline(
                y=overbought, color="r", linestyle="--", label="Overbought Threshold"
            )
            ax.axhline(
                y=oversold, color="g", linestyle="--", label="Oversold Threshold"
            )
            ax.set_title(f"{ticker} Z-Score")
            ax.legend()
            ax.grid(True)

            # Simplify x-axis: Show only start and end dates
            start_date = z_scores.index.min()
            end_date = z_scores.index.max()
            ax.set_xticks([start_date, end_date])
            ax.set_xticklabels(
                [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            )

        # Hide any unused subplots
        for j in range(num_current, rows * cols):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle(f"Z-Scores Grid - Page {page + 1}", fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
        plt.show()
