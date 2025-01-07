# tests/test_portfolio_utils.py

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.portfolio_utils import (
    convert_to_dict,
    normalize_weights,
    stacked_output,
    holdings_match,
)


class TestPortfolioUtils(unittest.TestCase):

    def test_convert_to_dict_ndarray(self):
        weights = np.array([0.3, 0.5, 0.2])
        asset_names = ["Asset_A", "Asset_B", "Asset_C"]
        expected = {"Asset_A": 0.3, "Asset_B": 0.5, "Asset_C": 0.2}
        result = convert_to_dict(weights, asset_names)
        self.assertEqual(result, expected)

    def test_convert_to_dict_ndarray_mismatch_length(self):
        weights = np.array([0.3, 0.5])
        asset_names = ["Asset_A", "Asset_B", "Asset_C"]
        with self.assertRaises(ValueError):
            convert_to_dict(weights, asset_names)

    def test_convert_to_dict_dataframe(self):
        data = {"Weights": [0.3, 0.5, 0.2]}
        asset_names = ["Asset_A", "Asset_B", "Asset_C"]
        df = pd.DataFrame(data, index=asset_names)
        expected = {"Asset_A": 0.3, "Asset_B": 0.5, "Asset_C": 0.2}
        result = convert_to_dict(df, asset_names)
        self.assertEqual(result, expected)

    def test_convert_to_dict_dict(self):
        weights = {"Asset_A": 0.3, "Asset_B": 0.5, "Asset_C": 0.2}
        asset_names = ["Asset_A", "Asset_B", "Asset_C"]
        result = convert_to_dict(weights, asset_names)
        self.assertEqual(result, weights)

    def test_convert_to_dict_invalid_type(self):
        weights = [0.3, 0.5, 0.2]  # List is unsupported
        asset_names = ["Asset_A", "Asset_B", "Asset_C"]
        with self.assertRaises(TypeError):
            convert_to_dict(weights, asset_names)

    def test_normalize_weights(self):
        weights = {"Asset_A": 0.3, "Asset_B": 0.5, "Asset_C": 0.2}
        min_weight = 0.25
        expected = {"Asset_A": 0.375, "Asset_B": 0.625}  # Asset_C is filtered out
        result = normalize_weights(weights, min_weight)
        self.assertEqual(result, expected)

    def test_normalize_weights_all_below_min(self):
        weights = {"Asset_A": 0.1, "Asset_B": 0.15}
        min_weight = 0.2
        with self.assertRaises(ValueError):
            normalize_weights(weights, min_weight)

    def test_stacked_output(self):
        stack_dict = {
            "model_a": {"Asset_A": 0.5, "Asset_B": 0.3},
            "model_b": {"Asset_A": 0.2, "Asset_B": 0.4, "Asset_C": 0.4},
            "model_c": {"Asset_A": 0.6},
        }
        expected = {
            "Asset_A": round((0.5 + 0.2 + 0.6) / 3, 3),  # 1.3 / 3 ≈ 0.433
            "Asset_B": round((0.3 + 0.4 + 0.0) / 3, 3),  # 0.7 / 3 ≈ 0.233
            "Asset_C": round((0.0 + 0.4 + 0.0) / 3, 3),  # 0.4 / 3 ≈ 0.133
        }
        result = stacked_output(stack_dict)
        self.assertEqual(result, expected)

    def test_stacked_output_empty(self):
        stack_dict = {}
        with self.assertRaises(ValueError):
            stacked_output(stack_dict)

    def test_holdings_match_true(self):
        cached_model_dict = {"Asset_A": 0.6, "Asset_B": 0.4, "Asset_C": 0.0}
        input_file_symbols = ["Asset_A", "Asset_B", "Asset_C"]
        result = holdings_match(cached_model_dict, input_file_symbols)
        self.assertTrue(result)

    def test_holdings_match_missing_in_input(self):
        cached_model_dict = {"Asset_A": 0.6, "Asset_B": 0.4, "Asset_C": 0.0}
        input_file_symbols = ["Asset_A", "Asset_B"]
        result = holdings_match(cached_model_dict, input_file_symbols, test_mode=True)
        self.assertFalse(result)

    def test_holdings_match_missing_in_cache(self):
        cached_model_dict = {"Asset_A": 0.6, "Asset_B": 0.4}
        input_file_symbols = ["Asset_A", "Asset_B", "Asset_C"]
        result = holdings_match(cached_model_dict, input_file_symbols, test_mode=True)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
