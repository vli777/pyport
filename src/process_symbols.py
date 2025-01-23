from datetime import datetime
from pathlib import Path
import pandas as pd
import pytz

from utils import logger
from utils.data_utils import flatten_columns, format_to_df_format, get_stock_data
from utils.date_utils import find_valid_trading_date, is_after_4pm_est


def load_or_download_symbol_data(symbol, start_date, end_date, data_path, download):
    """
    Load the symbol data from a Parquet file, or download if missing or outdated.
    Merges new data with existing data (appending instead of overwriting),
    properly flattens columns, and adjusts update_start to valid trading dates.
    """
    # Convert start_date and end_date to pd.Timestamp
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    pq_file = Path(data_path) / f"{symbol}.parquet"

    est = pytz.timezone("US/Eastern")
    now_est = datetime.now(est)
    today_ts = pd.Timestamp(now_est.date())

    # 1) If file doesn't exist, download full range
    if not pq_file.is_file():
        logger.info(
            f"No existing file for {symbol}. Downloading full history from {start_date} to {end_date}."
        )
        df_new = get_stock_data(symbol, start_date=start_ts, end_date=end_ts)
        df_new = flatten_columns(df_new, symbol)

        if df_new.empty:
            logger.warning(
                f"No data returned for {symbol} in range {start_ts} - {end_ts}."
            )
            return pd.DataFrame()

        # Save raw data to parquet
        df_new.to_parquet(pq_file)

        # Return formatted data (single column) for internal use
        return format_to_df_format(df_new, symbol)

    # 2) If after market close and we already have today's data, skip
    if is_after_4pm_est() and pq_file.is_file():
        df_existing = pd.read_parquet(pq_file)
        df_existing = flatten_columns(df_existing, symbol)
        if not df_existing.empty:
            last_date = df_existing.index.max()
            if last_date is not None and last_date >= today_ts:
                logger.info(
                    f"{symbol}: Already have today's data after market close, skipping."
                )
                return format_to_df_format(df_existing, symbol)

    # 3) Adjust end_ts if before market open
    if now_est.time() < datetime.strptime("09:30", "%H:%M").time():
        last_valid_ts = find_valid_trading_date(end_ts, tz=est, direction="backward")
        if last_valid_ts < start_ts:
            logger.warning(
                f"{symbol}: No valid trading days in [{start_ts}, {end_ts}]."
            )
            df_existing = pd.read_parquet(pq_file)
            df_existing = flatten_columns(df_existing, symbol)
            return (
                format_to_df_format(df_existing, symbol)
                if not df_existing.empty
                else pd.DataFrame()
            )
        effective_end_ts = min(end_ts, last_valid_ts)
    else:
        effective_end_ts = end_ts

    # If effective_end_ts falls on or after today, adjust it to the last valid trading day before today
    if effective_end_ts.normalize() >= today_ts.normalize():
        # Find the last valid trading day before today
        effective_end_ts = find_valid_trading_date(
            today_ts - pd.Timedelta(days=1), tz=est, direction="backward"
        )

    # 4) Read and flatten existing data
    df_existing = pd.read_parquet(pq_file)
    df_existing = flatten_columns(df_existing, symbol)
    last_date = df_existing.index.max() if not df_existing.empty else None

    # 5) Handle forced download: redownload everything
    if download:
        logger.info(f"{symbol}: Forced download from {start_ts} to {effective_end_ts}.")
        df_new = get_stock_data(symbol, start_date=start_ts, end_date=effective_end_ts)
        df_new = flatten_columns(df_new, symbol)

        if not df_new.empty:
            df_combined = (
                pd.concat([df_existing, df_new]).sort_index().drop_duplicates()
            )
            df_combined.to_parquet(pq_file)
            return format_to_df_format(df_combined, symbol)
        else:
            return format_to_df_format(df_existing, symbol)

    # 6) Update missing data if needed
    last_valid = find_valid_trading_date(effective_end_ts, tz=est, direction="backward")
    # Adjust effective_end_ts to the last valid trading day
    effective_end_ts = min(effective_end_ts, last_valid)

    if last_date is None or last_date < effective_end_ts:
        # Determine tentative start for update
        tentative_start = (last_date + pd.Timedelta(days=1)) if last_date else start_ts
        # Adjust tentative_start to the next valid trading date
        next_valid = find_valid_trading_date(
            tentative_start, tz=est, direction="forward"
        )

        # Use '>=' to catch empty or invalid ranges
        if next_valid >= effective_end_ts:
            logger.debug(
                f"{symbol}: Data already up-to-date. No valid trading days for update."
            )
            return format_to_df_format(df_existing, symbol)

        update_start = next_valid

        logger.info(
            f"{symbol}: Updating data from {update_start} to {effective_end_ts}."
        )
        df_new = get_stock_data(
            symbol, start_date=update_start, end_date=effective_end_ts
        )
        df_new = flatten_columns(df_new, symbol)

        if not df_new.empty:
            df_combined = (
                pd.concat([df_existing, df_new]).sort_index().drop_duplicates()
            )
            df_combined.to_parquet(pq_file)
            return format_to_df_format(df_combined, symbol)
        else:
            logger.info(f"{symbol}: No new data found for the update period.")
            # Mark the date as checked to avoid future redundant attempts
            empty_df = pd.DataFrame(index=[effective_end_ts])
            df_combined = (
                pd.concat([df_existing, empty_df]).sort_index().drop_duplicates()
            )
            df_combined.to_parquet(pq_file)
            return format_to_df_format(df_combined, symbol)
    else:
        # Already up-to-date
        return format_to_df_format(df_existing, symbol)


def process_symbols(symbols, start_date, end_date, data_path, download):
    """
    For each symbol:
      1) Load or download data (using yfinance)
      2) Flatten columns if needed
      3) Ensure columns are OHLC
      4) Convert to multi-level columns: Outer = ticker, Inner = price fields
      5) Join all symbols into a single DataFrame
      6) Return the combined DataFrame
    """
    df_all = pd.DataFrame()
    start_date_ts = pd.Timestamp(start_date)

    for sym in symbols:
        # Load or download the data just once
        df_sym = load_or_download_symbol_data(
            sym, start_date, end_date, data_path, download
        )
        if df_sym.empty:
            logger.warning(f"No data for {sym}, skipping.")
            continue

        # Drop duplicates in the index (if any)
        df_sym = df_sym.loc[~df_sym.index.duplicated(keep="first")]

        # Print columns to inspect structure
        # print(f"Symbol: {sym}")
        # print(df_sym.columns)
        # print(df_sym.head())

        # Ensure format_to_df_format retains OHLC columns
        df_sym = format_to_df_format(df_sym, sym)

        # Slice from the start_date onward
        if not df_sym.empty:
            earliest = df_sym.index.min()
            if start_date_ts < earliest:
                logger.info(
                    f"{sym} data starts at {earliest}. Not slicing older than that."
                )
            else:
                pos = df_sym.index.get_indexer([start_date_ts], method="nearest")[0]
                df_sym = df_sym.iloc[pos:]

        # Convert columns to a multi-level: Outer = ticker, Inner = original column name
        # Example final columns: (AAPL, 'Close'), (AAPL, 'High'), (AAPL, 'Adj Close'), etc.
        new_cols = pd.MultiIndex.from_product([[sym], df_sym.columns])
        df_sym.columns = new_cols

        # Join the symbol's DataFrame into the main df_all
        if df_all.empty:
            df_all = df_sym
        else:
            df_all = df_all.join(df_sym, how="outer")

    # Forward fill then backward fill to handle missing values
    df_all.ffill(inplace=True)
    df_all.bfill(inplace=True)

    if df_all.isna().any().any():
        logger.warning(
            "Data still has missing values after fill. Dropping remaining nulls."
        )
        df_all.dropna(inplace=True)

    return df_all
