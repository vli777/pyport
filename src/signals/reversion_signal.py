import pandas as pd

from signals.reversion_window import manage_dynamic_windows
from signals.dynamic_threshold import get_dynamic_thresholds
from signals.z_score import plot_z_scores_grid
from signals.optimize_multiplier import optimize_multiplier
from utils import logger


def generate_mean_reversion_signals(z_score_df, dynamic_thresholds):
    """
    Generate mean reversion signals based on Z-Score and dynamic thresholds.

    Args:
        z_score_df (pd.DataFrame): Z-Score DataFrame with tickers as columns and dates as index.
        dynamic_thresholds (dict): {ticker: (overbought_threshold, oversold_threshold)}

    Returns:
        dict: {ticker: {'overextended': bool, 'oversold': bool}}
    """
    signals = {}

    # Iterate over each ticker to generate signals
    for ticker in z_score_df.columns:
        if ticker not in dynamic_thresholds:
            continue

        # Get the latest Z-score and thresholds
        latest_z = z_score_df[ticker].iloc[-1]
        overbought_threshold, oversold_threshold = dynamic_thresholds[ticker]

        # Determine signals
        overextended = latest_z > overbought_threshold
        oversold = latest_z < oversold_threshold

        # Store results in dictionary
        signals[ticker] = {"overextended": overextended, "oversold": oversold}

    return signals


def apply_mean_reversion(
    returns_df,
    test_windows=range(10, 101, 10),  # Range of rolling windows to test
    plot=False,
    n_jobs=-1,  # Number of parallel jobs; -1 uses all available cores
):
    """
    Apply mean reversion strategy with dynamic windows and thresholds based on log returns.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame with tickers as columns and dates as index.
        test_windows (iterable, optional): Range of rolling windows to test during discovery.
        multiplier (float): Multiplier for threshold adjustment.
        plot (bool): Whether to plot Z-Scores for visualization.
        n_jobs (int): Number of parallel jobs to run. -1 utilizes all available CPU cores.

    Returns:
        Tuple[List[str], List[str]]:
            - List of ticker symbols to exclude (overbought).
            - List of ticker symbols to include (oversold).
    """
    logger.info("Discovering optimal rolling windows for each ticker...")
    dynamic_windows = manage_dynamic_windows(
        returns_df=returns_df,
        test_windows=test_windows,
        overbought_threshold=1.0,
        oversold_threshold=-1.0,
        n_jobs=n_jobs,
    )
    logger.info("Optimal rolling windows discovered.")

    # Calculate Z-scores and dynamic thresholds
    z_scores_dict = {}
    for ticker in returns_df.columns:
        window = dynamic_windows.get(ticker, 20)
        rolling_mean = returns_df[ticker].rolling(window).mean()
        rolling_std = returns_df[ticker].rolling(window).std()
        z_scores_dict[ticker] = (returns_df[ticker] - rolling_mean) / rolling_std

    z_score_df = pd.DataFrame(z_scores_dict)

    optimal_multipliers = optimize_multiplier(returns_df=returns_df, window=20)
    overbought_multiplier = optimal_multipliers["overbought_multiplier"]
    oversold_multiplier = optimal_multipliers["oversold_multiplier"]
    logger.info(
        f"Optimal multipliers determined: {overbought_multiplier}, {oversold_multiplier}"
    )

    dynamic_thresholds = get_dynamic_thresholds(
        returns_df,
        window=20,
        overbought_multiplier=overbought_multiplier,
        oversold_multiplier=oversold_multiplier,
    )

    # Generate signals
    signals = generate_mean_reversion_signals(z_score_df, dynamic_thresholds)

    # Separate tickers based on signals
    tickers_to_exclude = [
        ticker for ticker, signal in signals.items() if signal["overextended"]
    ]
    tickers_to_include = [
        ticker for ticker, signal in signals.items() if signal["oversold"]
    ]

    logger.info(f"MR Exclusions (Overbought): {tickers_to_exclude}")
    logger.info(f"MR Inclusions (Oversold): {tickers_to_include}")

    # Optional: Plot Z-Scores
    if plot:
        plot_z_scores_grid(
            z_scores_df=z_score_df,
            overbought_thresholds=pd.Series(
                {k: v[0] for k, v in dynamic_thresholds.items()}
            ),
            oversold_thresholds=pd.Series(
                {k: v[1] for k, v in dynamic_thresholds.items()}
            ),
            grid_shape=(6, 6),
            figsize=(24, 24),
        )

    return tickers_to_exclude, tickers_to_include
