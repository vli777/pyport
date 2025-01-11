# tests/test_date_utils.py

import unittest
from datetime import datetime, date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src.utils.date_utils import (
    str_to_date,
    date_to_str,
    is_weekday,
    is_after_4pm_est,
    get_non_holiday_weekdays,
    calculate_start_end_dates,
    get_last_date,
)
from unittest.mock import patch
import pytz
from pathlib import Path
import csv


class TestDateUtils(unittest.TestCase):

    def test_str_to_date_valid(self):
        self.assertEqual(str_to_date("2023-10-05"), datetime(2023, 10, 5))

    def test_str_to_date_invalid(self):
        with self.assertRaises(ValueError):
            str_to_date("2023-13-01")

    def test_date_to_str(self):
        dt = datetime(2023, 10, 5)
        self.assertEqual(date_to_str(dt), "2023-10-05")

    def test_is_weekday_true(self):
        self.assertTrue(is_weekday(date(2023, 10, 5)))  # Thursday

    def test_is_weekday_false(self):
        self.assertFalse(is_weekday(date(2023, 10, 7)))  # Saturday

    @patch("utils.date_utils.datetime")
    def test_is_after_4pm_est_true(self, mock_datetime):
        est = pytz.timezone("US/Eastern")
        mock_now = est.localize(datetime(2023, 10, 5, 16, 1)) 
        mock_datetime.now.return_value = mock_now
        self.assertTrue(is_after_4pm_est())

    @patch("utils.date_utils.datetime")
    def test_is_after_4pm_est_false(self, mock_datetime):
        est = pytz.timezone("US/Eastern")
        mock_now = datetime(2023, 10, 5, 15, 59, tzinfo=est)
        mock_datetime.now.return_value = mock_now
        self.assertFalse(is_after_4pm_est())

    def test_get_non_holiday_weekdays(self):
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 10)
        first, last = get_non_holiday_weekdays(start_date, end_date)
        self.assertEqual(
            first, date(2023, 1, 3)
        )  # First NYSE open day after Jan 1, 2023
        self.assertEqual(
            last, date(2023, 1, 10)
        )  # Last NYSE open day before Jan 10, 2023

    def test_calculate_start_end_dates(self):
        reference_date = date(2023, 10, 5)
        start, end = calculate_start_end_dates(1, reference_date)
        expected_start = date(2022, 10, 5)
        # Adjust expected_start and end based on NYSE schedule if needed
        self.assertEqual(start, expected_start)
        self.assertEqual(end, reference_date)

    def test_get_last_date_valid(self):
        # Create a temporary CSV file
        temp_csv = Path("temp_test.csv")
        with temp_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["2023-10-01", "data1"])
            writer.writerow(["2023-10-02", "data2"])
            writer.writerow(["2023-10-03", "data3"])

        self.assertEqual(get_last_date("temp_test.csv"), date(2023, 10, 3))
        temp_csv.unlink()  # Clean up

    def test_get_last_date_invalid_format(self):
        temp_csv = Path("temp_test_invalid.csv")
        with temp_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["invalid-date", "data1"])
            writer.writerow(["2023-10-02", "data2"])

        self.assertEqual(get_last_date("temp_test_invalid.csv"), date(2023, 10, 2))
        temp_csv.unlink()

    def test_get_last_date_no_valid_dates(self):
        temp_csv = Path("temp_test_no_valid.csv")
        with temp_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["invalid-date", "data1"])
            writer.writerow(["another-invalid", "data2"])

        self.assertIsNone(get_last_date("temp_test_no_valid.csv"))
        temp_csv.unlink()

    def test_get_last_date_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_last_date("non_existent_file.csv")


if __name__ == "__main__":
    unittest.main()
