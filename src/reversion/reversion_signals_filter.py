from typing import List, Tuple, Dict
import pandas as pd


def generate_mean_reversion_signals(
    z_score_df: pd.DataFrame, dynamic_thresholds: Dict[str, Tuple[float, float]]
) -> Dict[str, Dict[str, bool]]:
    """
    Generate mean reversion signals based on Z-Score and dynamic thresholds.

    Args:
        z_score_df (pd.DataFrame): Z-Score DataFrame with tickers as columns and dates as index.
        dynamic_thresholds (dict): {ticker: (overbought_threshold, oversold_threshold)}

    Returns:
        dict: {ticker: {'overextended': bool, 'oversold': bool}}
    """
    # Extract thresholds into separate Series for vectorized comparisons
    overbought = pd.Series({ticker: thresholds[0] for ticker, thresholds in dynamic_thresholds.items()})
    oversold = pd.Series({ticker: thresholds[1] for ticker, thresholds in dynamic_thresholds.items()})
    
    # Align the thresholds with the DataFrame
    valid_tickers = z_score_df.columns.intersection(dynamic_thresholds.keys())
    z_score_subset = z_score_df[valid_tickers].iloc[-1]  # Latest Z-scores

    # Generate signals using vectorized comparisons
    overextended = z_score_subset > overbought
    oversold_signals = z_score_subset < oversold

    # Combine into a DataFrame
    signals = pd.DataFrame({
        "overextended": overextended,
        "oversold": oversold_signals
    })

    return signals
