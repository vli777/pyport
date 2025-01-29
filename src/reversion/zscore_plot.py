from typing import Dict, Tuple
import matplotlib.pyplot as plt
import math
import pandas as pd


def plot_z_scores_grid(
    z_scores_df: pd.DataFrame,
    dynamic_thresholds: Dict[
        str, Tuple[float, float]
    ],  # Directly use the dict from get_dynamic_thresholds
    grid_shape=(6, 6),
    figsize=(20, 20),
):
    """
    Plot Z-Scores for multiple tickers in a paginated grid layout.

    Args:
        z_scores_df (pd.DataFrame): DataFrame containing Z-Scores for all tickers.
                                     Columns are tickers, index are dates.
        dynamic_thresholds (Dict[str, Tuple[float, float]]): Dictionary of overbought/oversold thresholds.
                                                             Keys should match z_scores_df columns.
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

        fig, axes = plt.subplots(*grid_shape, figsize=figsize)
        axes = axes.flatten()  # Flatten to 1D for easy iteration

        for i, ticker in enumerate(current_tickers):
            ax = axes[i]
            z_scores = z_scores_df[ticker]

            # Retrieve dynamic thresholds
            overbought, oversold = dynamic_thresholds.get(ticker, (2.0, -2.0))

            ax.plot(z_scores.index, z_scores, label=f"{ticker} Z-Score", color="blue")
            ax.axhline(y=overbought, color="r", linestyle="--", label="Overbought")
            ax.axhline(y=oversold, color="g", linestyle="--", label="Oversold")
            ax.set_title(f"{ticker}")
            ax.legend()
            ax.grid(True)

            # Set only start and end date ticks
            start_date = z_scores.index.min()
            end_date = z_scores.index.max()
            ax.set_xticks([start_date, end_date])
            ax.set_xticklabels(
                [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            )

        # Hide any unused subplots
        for j in range(len(current_tickers), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle(f"Z-Scores Grid - Page {page + 1}", fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
        plt.show()
