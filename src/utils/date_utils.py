# src/utils/date_utils.py

from datetime import datetime, date
from pathlib import Path
import pandas_market_calendars as mcal
import pytz
import csv
from typing import Tuple, Optional
from dateutil.relativedelta import relativedelta
from .logger import logger


def str_to_date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime:
    """
    Convert a string to a datetime object.

    Args:
        date_str (str): The date string to convert.
        fmt (str, optional): The format of the date string. Defaults to "%Y-%m-%d".

    Returns:
        datetime: The corresponding datetime object.
    """
    try:
        return datetime.strptime(date_str, fmt)
    except ValueError as e:
        raise ValueError(f"Error parsing date string '{date_str}': {e}") from e


def date_to_str(date_obj: datetime, fmt: str = "%Y-%m-%d") -> str:
    """
    Convert a datetime object to a string.

    Args:
        date_obj (datetime): The datetime object to convert.
        fmt (str, optional): The desired string format. Defaults to "%Y-%m-%d".

    Returns:
        str: The formatted date string.
    """
    return date_obj.strftime(fmt)


def is_weekday(date_obj: date) -> bool:
    """
    Check if a given date is a weekday.

    Args:
        date_obj (date): The date to check.

    Returns:
        bool: True if the date is a weekday (Monday to Friday), False otherwise.
    """
    return date_obj.weekday() < 5  # Monday to Friday are weekdays (0 to 4)


def is_after_4pm_est(current_time: Optional[datetime] = None) -> bool:
    """
    Determine if the current time is after 4:01 PM EST.

    Args:
        current_time (Optional[datetime], optional): The current time. If None, uses the current system time. Defaults to None.

    Returns:
        bool: True if after 4:01 PM EST, False otherwise.
    """
    est = pytz.timezone("US/Eastern")
    now = current_time.astimezone(est) if current_time else datetime.now(est)
    return now.hour > 16 or (now.hour == 16 and now.minute >= 1)


def get_non_holiday_weekdays(
    start_date: date, end_date: date, tz: pytz.timezone = pytz.timezone("US/Eastern")
) -> Tuple[date, date]:
    """
    Retrieve the first and last non-holiday weekdays within a date range based on the NYSE calendar.

    Args:
        start_date (date): The start date.
        end_date (date): The end date.
        tz (pytz.timezone, optional): The timezone. Defaults to US/Eastern.

    Returns:
        Tuple[date, date]: A tuple containing the first and last valid dates.
    """
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    if schedule.empty:
        return start_date, start_date

    first_date = schedule.index[0].date()
    last_date = schedule.index[-1].date()
    return first_date, last_date


def calculate_start_end_dates(
    years: int, reference_date: Optional[date] = None
) -> Tuple[date, date]:
    """
    Calculate the start and end dates based on the number of years from a reference date.

    Args:
        years (int): Number of years to subtract from the reference date.
        reference_date (Optional[date], optional): The reference date. Defaults to today's date.

    Returns:
        Tuple[date, date]: A tuple containing the start and end dates.
    """
    reference_date = reference_date or datetime.now().date()
    start_date_from_ref = reference_date - relativedelta(years=years)
    return get_non_holiday_weekdays(start_date_from_ref, reference_date)


def get_last_date(csv_filename: str) -> Optional[date]:
    """
    Retrieve the last valid date from a CSV file where dates are in the first column.

    This function reads the file in reverse to efficiently find the last valid date.

    Args:
        csv_filename (str): The path to the CSV file.

    Returns:
        Optional[date]: The last valid date found, or None if not found.
    """
    csv_path = Path(csv_filename)

    if not csv_path.is_file():
        logger.error(f"File '{csv_filename}' does not exist.")
        raise FileNotFoundError(f"File '{csv_filename}' does not exist.")

    try:
        with csv_path.open("r", newline="") as file:
            reader = csv.reader(file)
            for row in reversed(list(reader)):
                if row:
                    last_date_str = row[0].strip()
                    try:
                        return datetime.strptime(last_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        logger.warning(
                            f"Invalid date format: '{last_date_str}' in file '{csv_filename}'. Skipping."
                        )
        logger.info(f"No valid dates found in '{csv_filename}'.")
        return None
    except Exception as e:
        logger.exception(f"An error occurred while reading '{csv_filename}': {e}")
        raise
