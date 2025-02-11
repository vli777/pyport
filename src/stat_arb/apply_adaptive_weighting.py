import numpy as np
import pandas as pd


def apply_adaptive_weighting(
    baseline_allocation: dict,
    mean_reversion_weights: pd.Series,
    returns_df: pd.DataFrame,
    base_alpha: float = 0.2,
    allow_short: bool = False,
) -> pd.Series:
    """
    Applies mean reversion-based adaptive weighting to a baseline allocation.

    Args:
        baseline_allocation (dict): Baseline portfolio weights from model optimization.
        mean_reversion_weights (pd.Series): Mean reversion suggested allocations.
        returns_df (pd.DataFrame): Historical returns to compute volatility.
        base_alpha (float): Sensitivity factor for mean reversion adjustment.
        allow_short (bool): If False, prevents negative allocations.

    Returns:
        pd.Series: Adjusted portfolio allocation.
    """
    # Convert baseline_allocation from dict to Pandas Series
    baseline_allocation = pd.Series(baseline_allocation, dtype=float)

    # Ensure both Series have the same index (tickers)
    baseline_allocation = baseline_allocation.reindex(
        mean_reversion_weights.index, fill_value=0
    )

    # Compute realized volatility (prevent division by zero)
    realized_volatility = (
        returns_df.rolling(window=30, min_periods=5).std().mean(axis=1).iloc[-1]
    )
    realized_volatility = max(realized_volatility, 1e-6)  # Avoid division by zero

    adaptive_alpha = base_alpha / (1 + realized_volatility)

    # Prevent division by zero for baseline_allocation
    safe_baseline = baseline_allocation.replace(
        0, np.nan
    )  # Replace 0s with NaN to avoid div-by-zero
    adjustment_factor = 1 + adaptive_alpha * (
        (mean_reversion_weights / safe_baseline) - 1
    )
    adjustment_factor = adjustment_factor.fillna(
        1
    )  # Replace NaNs with 1 (no change for zero weights)

    # Apply the adjusted weights
    adjusted_allocation = baseline_allocation * adjustment_factor

    # Handle negative weights if shorting is disabled
    if not allow_short:
        adjusted_allocation = adjusted_allocation.clip(lower=0)

    # Normalize the final weights to sum to 1 if non-zero
    total_weight = adjusted_allocation.sum()
    if total_weight > 0:
        adjusted_allocation /= total_weight

    return adjusted_allocation
