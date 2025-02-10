from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class CointegrationAnalyzer:
    def __init__(self, prices_df: pd.DataFrame, det_order=0, k_ar_diff=1):
        """
        Analyzes multi-asset cointegration and computes spreads for basket trading.

        Args:
            prices_df (pd.DataFrame): Log-prices DataFrame with dates as index, assets as columns.
            det_order (int): Deterministic trend order for Johansen test.
            k_ar_diff (int): Number of lag differences.
        """
        self.prices_df = prices_df
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff
        self.cointegration_vector = self._compute_cointegration_vector()
        self.spread_series = self._compute_basket_spread()

    def _compute_cointegration_vector(self):
        """
        Runs Johansen's test and extracts the first cointegrating vector.
        """
        if self.prices_df.shape[1] < 2:
            raise ValueError("Johansen test requires at least two asset price series.")

        result = coint_johansen(self.prices_df, self.det_order, self.k_ar_diff)
        return result.evec[:, 0]  # First eigenvector

    def _compute_basket_spread(self):
        """
        Computes the spread using the cointegration vector.
        """
        return self.prices_df.dot(self.cointegration_vector)

    def detect_outlier(self, contamination: float = 0.05) -> int:
        """
        Uses Isolation Forest to detect if the latest spread is an outlier.

        Args:
            contamination (float): Fraction of expected outliers.

        Returns:
            int: 1 if normal, -1 if outlier.
        """
        spread_series = self.spread_series.dropna()
        if len(spread_series) < 2:
            return 1  # Not enough data to determine outliers, assume inlier
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(spread_series.values.reshape(-1, 1))
        flag: int = iso.predict(np.array([[spread_series.iloc[-1]]]))[0]
        return flag

    def get_hedge_ratios(self):
        """
        Returns hedge ratios normalized to sum to 1.
        """
        return self.cointegration_vector / np.abs(self.cointegration_vector).sum()
