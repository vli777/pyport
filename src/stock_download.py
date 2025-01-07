import re
from pathlib import Path
import yfinance as yf
import csv
import pandas as pd


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
                            print(f"Ignoring invalid ticker symbol: {line}")
        except FileNotFoundError:
            print(f"File not found: {input_file}")
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")

    return sorted(symbols)


def get_stock_data(symbol, start_date, end_date):
    """
    download stock price data from yahoo finance with optional write to csv
    """
    print(f"Downloading {symbol} {start_date} - {end_date} ...")
    symbol_df = yf.download(symbol, start=start_date, end=end_date)
    return symbol_df


def update_store(data_path, symbol, symbol_df, start_date, end_date):
    """
    Append new stock data to the existing CSV file and update the in-memory DataFrame,
    avoiding duplicate entries.
    """
    new_data = get_stock_data(symbol, start_date=start_date, end_date=end_date)
    csv_filename = Path(data_path) / f"{symbol}.csv"

    # Make sure there is no overlap between new data and existing data
    # Remove rows from new_data that are already in symbol_df
    new_data = new_data.loc[~new_data.index.isin(symbol_df.index)]

    # If new data is available (i.e., no overlap with existing data)
    if not new_data.empty:
        # Append new data to the CSV file
        with csv_filename.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for index, row in new_data.iterrows():
                index_date_string = index.strftime("%Y-%m-%d")
                writer.writerow([index_date_string] + row.tolist())

        # Update symbol_df with new data, removing duplicates (if any)
        symbol_df = pd.concat([symbol_df, new_data], axis=0).drop_duplicates()

    return symbol_df
