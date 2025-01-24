import matplotlib.pyplot as plt
import math


def calculate_z_score(price_series, window):
    """
    Calculate the Z-Score for a given price series based on a rolling window.

    Args:
        price_series (pd.Series): Series of 'Adj Close' prices for a ticker.
        window (int): Rolling window size.

    Returns:
        pd.Series: Z-Score series.
    """
    rolling_mean = price_series.rolling(window=window, min_periods=1).mean()
    rolling_std = price_series.rolling(window=window, min_periods=1).std()
    z_scores = (price_series - rolling_mean) / rolling_std
    return z_scores


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

        # Hide any unused subplots
        for j in range(num_current, rows * cols):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle(f"Z-Scores Grid - Page {page + 1}", fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
        plt.show()
