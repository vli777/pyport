# tests/test_caching_utils.py

import sys
from pathlib import Path
import unittest
import json
import os
from datetime import datetime, timedelta

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.caching_utils import (
    save_model_results,
    load_model_results_from_cache,
    cleanup_cache,
)
from utils.logger import logger


class TestCachingUtils(unittest.TestCase):

    def setUp(self):
        """Set up a temporary cache directory and test data."""
        self.cache_dir = Path.cwd() / "cache_test"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = "test_model"
        self.time_period = "monthly"
        self.input_filename = "input_test"
        self.symbols = ["Asset_A", "Asset_B", "Asset_C"]
        self.scaled = {"Asset_A": 0.3, "Asset_B": 0.5}
        self.expected_results = {"Asset_A": 0.3, "Asset_B": 0.5, "Asset_C": 0.0}
        self.cache_file = (
            self.cache_dir
            / f"{self.input_filename}-{self.model_name}-{self.time_period}.json"
        )

    def tearDown(self):
        """Clean up the temporary cache directory after tests."""
        if self.cache_dir.exists():
            for file in self.cache_dir.iterdir():
                file.unlink()
            self.cache_dir.rmdir()

    def test_save_model_results(self):
        """Test saving model results to cache."""
        save_model_results(
            model_name=self.model_name,
            time_period=self.time_period,
            input_filename=self.input_filename,
            symbols=self.symbols,
            scaled=self.scaled,
            cache_dir=self.cache_dir,
        )

        self.assertTrue(self.cache_file.exists(), "Cache file was not created.")

        with self.cache_file.open("r") as jsonfile:
            data = json.load(jsonfile)

        self.assertEqual(
            data, self.expected_results, "Cached data does not match expected results."
        )

    def test_load_model_results_from_cache_valid(self):
        """Test loading valid model results from cache."""
        # First, save the results
        save_model_results(
            model_name=self.model_name,
            time_period=self.time_period,
            input_filename=self.input_filename,
            symbols=self.symbols,
            scaled=self.scaled,
            cache_dir=self.cache_dir,
        )

        # Now, load the results
        loaded_results = load_model_results_from_cache(
            model_name=self.model_name,
            time_period=self.time_period,
            input_filename=self.input_filename,
            symbols=self.symbols,
            cache_dir=self.cache_dir,
        )

        self.assertIsNotNone(loaded_results, "Failed to load valid cache.")
        self.assertEqual(
            loaded_results,
            self.expected_results,
            "Loaded cache data does not match expected.",
        )

    def test_load_model_results_from_cache_invalid_symbols(self):
        """Test loading cache with mismatched symbols."""
        # Save the results
        save_model_results(
            model_name=self.model_name,
            time_period=self.time_period,
            input_filename=self.input_filename,
            symbols=self.symbols,
            scaled=self.scaled,
            cache_dir=self.cache_dir,
        )

        # Modify symbols to simulate mismatch
        new_symbols = ["Asset_A", "Asset_B", "Asset_D"]

        # Attempt to load with mismatched symbols
        loaded_results = load_model_results_from_cache(
            model_name=self.model_name,
            time_period=self.time_period,
            input_filename=self.input_filename,
            symbols=new_symbols,
            cache_dir=self.cache_dir,
        )

        self.assertIsNone(
            loaded_results, "Loaded cache should be None due to symbol mismatch."
        )

    def test_load_model_results_from_cache_nonexistent_file(self):
        """Test loading cache when the cache file does not exist."""
        loaded_results = load_model_results_from_cache(
            model_name=self.model_name,
            time_period=self.time_period,
            input_filename=self.input_filename,
            symbols=self.symbols,
            cache_dir=self.cache_dir,
        )

        self.assertIsNone(
            loaded_results,
            "Loaded cache should be None when cache file does not exist.",
        )

    # tests/test_caching_utils.py

    def test_cleanup_cache(self):
        """Test cleaning up old cache files."""
        # Save a cache file with an old modification time
        old_cache_file = (
            self.cache_dir
            / f"{self.input_filename}-{self.model_name}-{self.time_period}.json"
        )
        with old_cache_file.open("w") as jsonfile:
            json.dump(self.expected_results, jsonfile)

        # Modify the modification time to be older than max_age_hours
        old_time = datetime.now() - timedelta(hours=25)
        old_timestamp = old_time.timestamp()
        try:
            os.utime(old_cache_file, (old_timestamp, old_timestamp))
        except Exception as e:
            logger.error(f"Failed to modify file time for {old_cache_file}: {e}")
            self.fail(f"Failed to modify file time for {old_cache_file}: {e}")

        # Ensure the file exists before cleanup
        self.assertTrue(
            old_cache_file.exists(), "Old cache file does not exist before cleanup."
        )

        # Perform cleanup with max_age_hours=24
        cleanup_cache(cache_dir=str(self.cache_dir), max_age_hours=24)

        # Check that the old cache file has been removed
        self.assertFalse(
            old_cache_file.exists(), "Old cache file was not removed during cleanup."
        )

    def test_cleanup_cache_nonexistent_directory(self):
        """Test cleaning up when the cache directory does not exist."""
        non_existent_dir = self.cache_dir / "nonexistent"
        try:
            cleanup_cache(cache_dir=str(non_existent_dir), max_age_hours=24)
            # If no exception is raised, the test passes
            self.assertTrue(
                True, "No exception raised when cleaning a nonexistent directory."
            )
        except Exception as e:
            self.fail(
                f"cleanup_cache raised an exception for a nonexistent directory: {e}"
            )


if __name__ == "__main__":
    unittest.main()
