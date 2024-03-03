import re
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
    symbols = set()
    for input_file in input_file_paths:
        with open(input_file + ".csv", 'r') as file:
            for line in file:
                line = line.strip().upper()
                if re.match("^[A-Z]+$", line) and not line.startswith("#"):
                    if is_valid_ticker(line):
                        symbols.add(line)
                    else:
                        print(f"Ignoring invalid ticker symbol: {line}")
    return sorted(symbols)

def get_stock_data(symbol, start_date, end_date):
    """
    download stock price data from yahoo finance with optional write to csv
    """
    print(f"Downloading {symbol} {start_date} - {end_date} ...")
    symbol_df = yf.download(symbol, start=start_date, end=end_date)
    return symbol_df

def update_store(data_path, symbol, symbol_df, start_date, end_date):
    new_data = get_stock_data(symbol, start_date=start_date, end_date=end_date)
    
    # Append new data to the CSV file
    csv_filename = data_path + f"{symbol}.csv"
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write each row with the index included
        for index, row in new_data.iterrows():
            index_date_string = index.strftime('%Y-%m-%d')  # Convert index to date string
            writer.writerow([index_date_string] + row.tolist())

    # Update symbol_df with new data
    symbol_df = pd.concat([symbol_df, new_data], axis=0)    
    return symbol_df