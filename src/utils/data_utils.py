import pandas as pd
import yfinance as yf
from pathlib import Path
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from logger import logger


def is_valid_ticker(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return True
    except:
        return False


def process_input_files(input_file_paths):
    """
    Process input CSV files, read valid ticker symbols, and return a sorted list of unique symbols.
    Symbols are validated using a regular expression and a custom function `is_valid_ticker`.
    """
    symbols = set()
    for input_file in input_file_paths:
        input_file = Path(input_file).with_suffix(
            ".csv"
        )  # Ensure the file has a .csv extension

        try:
            with input_file.open("r") as file:
                for line in file:
                    line = line.strip().upper()
                    # Validate ticker: must be alphabetic and not start with a comment ('#')
                    if re.match("^[A-Z]+$", line) and not line.startswith("#"):
                        if is_valid_ticker(line):
                            symbols.add(line)
                        else:
                            logger.warning(f"Ignoring invalid ticker symbol: {line}")
        except FileNotFoundError:
            logger.warning(f"File not found: {input_file}")
        except Exception as e:
            logger.warning(f"Error processing file {input_file}: {e}")

    return sorted(symbols)


def get_stock_data(symbol, start_date, end_date):
    """
    download stock price data from yahoo finance with optional write to csv
    """
    logger.info(f"Downloading {symbol} {start_date} - {end_date} ...")
    symbol_df = yf.download(symbol, start=start_date, end=end_date)
    return symbol_df


def update_store(data_path, symbol, start_date, end_date):
    """
    Reads existing data for `symbol` from a Parquet file (if present).
    Fetches new data from Yahoo Finance for the date range.
    Merges old and new data, drops duplicates.
    Writes the combined dataset back to a Parquet file.

    Returns:
        pd.DataFrame: The updated DataFrame (in memory).
    """
    # 1) Locate the Parquet file
    pq_filename = Path(data_path) / f"{symbol}.parquet"

    # 2) Load existing data if file exists
    if pq_filename.is_file():
        old_df = pd.read_parquet(pq_filename)
    else:
        old_df = pd.DataFrame()

    # 3) Download new data
    logger.info(f"Downloading {symbol} {start_date} - {end_date} ...")
    new_df = yf.download(symbol, start=start_date, end=end_date)

    if not new_df.empty:
        # If the index is not DateTime, convert it
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)

        # 4) Merge old and new data
        combined_df = pd.concat([old_df, new_df]).sort_index().drop_duplicates()

        # 5) Write the entire dataset back in one shot
        combined_df.to_parquet(pq_filename, index=True)

        return combined_df
    else:
        logger.info(f"No new data found for {symbol}. Skipping update.")
        return old_df


def download_all_tickers(watchlist_files, data_path, years=5):
    """
    1) Collect tickers from watchlist CSV files.
    2) For each ticker, determine last date or fallback to X years ago.
    3) Download/append to a Parquet file via update_store_parquet().
    """
    symbols = process_input_files(watchlist_files)  # Same as your CSV-based code
    today = datetime.now().date()

    for symbol in symbols:
        pq_filename = Path(data_path) / f"{symbol}.parquet"

        # Check if parquet file exists
        if pq_filename.is_file():
            # Attempt to read last date from the existing Parquet
            last_date = get_last_date(pq_filename)
            if last_date:
                start_date = last_date + timedelta(days=1)
            else:
                # If file has no valid data
                start_date = today - relativedelta(years=years)
        else:
            # No existing file => fetch from X years ago
            start_date = today - relativedelta(years=years)

        # Only download if we actually need data
        if start_date <= today:
            update_store(data_path, symbol, start_date, today)
        else:
            logger.info(f"No update needed for {symbol} — up to date.")


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


def convert_all_csv_to_parquet(data_folder: str):
    """
    Convert all CSV files in `data_folder` to Parquet.
    Assumes the first column of each CSV is the date column and should be the index.
    """
    data_path = Path(data_folder)

    # Loop through every CSV file in data_path
    for csv_file in data_path.glob("*.csv"):
        logger.info(f"Converting {csv_file} to Parquet...")

        # Read CSV, parse first column as dates, set as index
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)

        # Optionally ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Construct the Parquet file path (same base name, different extension)
        parquet_file = csv_file.with_suffix(".parquet")

        # Write DataFrame to Parquet
        df.to_parquet(parquet_file)

        logger.info(f"Converted {csv_file.name} -> {parquet_file.name}")
