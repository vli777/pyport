from datetime import datetime, timedelta
from pathlib import Path
import pytz
import trading_calendars as tc
import pandas as pd

def cleanup_cache(cache_dir, max_age_hours=24):
    """
    Removes files in the cache directory that are older than max_age_hours.
    """
    now = datetime.now()
    max_age = timedelta(hours=max_age_hours)

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        print(f"Cache directory {cache_dir} does not exist.")
        return

    for filepath in cache_dir.iterdir():
        if filepath.is_file():
            creation_time = datetime.fromtimestamp(filepath.stat().st_ctime)
            if now - creation_time > max_age:
                print(f"Removing {filepath} (created at {creation_time})")
                filepath.unlink()

def str_to_date(date_str, fmt="%Y-%m-%d"):
    """
    convert string to datetime
    """
    return datetime.strptime(date_str, fmt)


def date_to_str(date, fmt="%Y-%m-%d"):
    """
    convert datetime to string
    """
    return date.strftime(fmt)

def is_weekday(date):
    return date.weekday() < 5  # Monday to Friday are weekdays (0 to 4)

def is_after_4pm_est():
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    return now.hour >= 16  # Check if it's after 4 PM EST

def get_non_holiday_weekdays(start_date, end_date, tz=pytz.timezone('US/Eastern')):            
    cal = tc.get_calendar("XNYS")    
    open_sessions = cal.sessions_in_range(pd.Timestamp(start_date, tz=tz), pd.Timestamp(end_date,tz=tz))
        
    if open_sessions.empty:
        return start_date, start_date
    
    first_date, last_date = open_sessions[0].date(), open_sessions[-1].date() + timedelta(days=1)
    return first_date, last_date

def calculate_start_end_dates(time):
    # Get today's date
    today = datetime.now().date()
    start_date_from_today = today - timedelta(days=time * 365)
    start_date, end_date = get_non_holiday_weekdays(start_date_from_today, today)    
    # Return calculated dates
    return start_date, end_date

def get_last_date(csv_filename):
    """
    Retrieve the last date from a CSV file where dates are in the first column.
    """
    csv_path = Path(csv_filename)
    if not csv_path.is_file():
        print(f"File {csv_filename} does not exist.")
        return None

    with csv_path.open('r') as file:
        lines = file.readlines()

        # Iterate over lines in reverse order to find the last valid date
        for line in reversed(lines):
            line = line.strip()
            if line:  # Check if the line is not empty
                last_date_str = line.split(',')[0]  # Assuming date is the first field
                try:
                    return datetime.strptime(last_date_str, '%Y-%m-%d').date()
                except ValueError:
                    print(f"Invalid date format: {last_date_str}")
                    continue  # Continue if date format is invalid
    return None
