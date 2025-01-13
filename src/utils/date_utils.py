# src/utils/date_utils.py

from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import pandas_market_calendars as mcal
import pytz
from typing import Tuple, Optional

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
    years: float = 1.0,
    reference_date: Optional[date] = None,
) -> Tuple[date, date]:
    """
    Calculate the start and end dates based on the number of years from a reference date.

    Args:
        years (float): Number of years to subtract from the reference date.
        reference_date (Optional[date], optional): The reference date. Defaults to today's date.

    Returns:
        Tuple[date, date]: A tuple containing the start and end dates.
    """
    if reference_date is None:
        reference_date = date.today()

    # Convert fractional years to days
    days = int(round(years * 365))
    start_date = reference_date - timedelta(days=days)

    # Adjust dates for non-holiday weekdays
    return get_non_holiday_weekdays(start_date, reference_date)


def get_last_date(parquet_file: Path):
    """
    Reads the Parquet file, inspects the last row, and returns that date as a `date` object.
    """
    try:
        df = pd.read_parquet(parquet_file)
        if not df.empty:
            # Assuming the index is a DatetimeIndex
            return df.index.max().date()
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not read Parquet file {parquet_file}: {e}")
        return None


def find_valid_trading_date(start_date, tz, direction='forward'):
    """
    Walk through time from 'start_date' in the specified direction until 
    a valid trading day is found. Returns a pandas Timestamp of that day.

    Args:
        start_date (datetime or date-like): The starting point for the search.
        tz (pytz.timezone): The timezone to consider for holiday/weekend calculations.
        direction (str): 'forward' to search future dates, 'backward' to search past dates.
                         Default is 'forward'.
    """
    current_date = pd.Timestamp(start_date).normalize()
    
    # Determine the step based on search direction
    if direction == 'forward':
        step = timedelta(days=1)
        weekday_check = lambda d: not is_weekday(d)  # Move forward if weekend
    elif direction == 'backward':
        step = -timedelta(days=1)
        weekday_check = lambda d: not is_weekday(d)  # Move backward if weekend
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    while True:
        # 1) If it's a weekend, step in the chosen direction
        if weekday_check(current_date.date()):
            current_date += step
            continue

        # 2) If it's a holiday, step in the chosen direction
        first_valid_date, _ = get_non_holiday_weekdays(
            current_date.date(), current_date.date(), tz=tz
        )
        if current_date.date() != first_valid_date:
            current_date += step
            continue

        # If we pass both checks, this is a valid trading day
        return current_date
