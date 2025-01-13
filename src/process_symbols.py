from datetime import datetime
from pathlib import Path
import pandas as pd
import pytz

from utils import logger
from utils.data_utils import format_parquet_columns, get_stock_data
from utils.date_utils import find_last_valid_trading_date, is_after_4pm_est


def load_or_download_symbol_data(symbol, start_date, end_date, data_path, download):
    """
    Load the symbol data from a Parquet file, or download if missing or outdated.
    If it's before market open, fetch data only up to the last valid trading day.
    If after market close and we already have 'today', skip further downloading.
    """
    # Convert start_date and end_date to pd.Timestamp
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    pq_file = Path(data_path) / f"{symbol}.parquet"
    df_sym = pd.DataFrame()

    est = pytz.timezone("US/Eastern")
    now_est = datetime.now(est)
    today_ts = pd.Timestamp(now_est.date())

    if not pq_file.is_file():
        logger.info(
            f"No existing file for {symbol}. Downloading full history from {start_date} to {end_date}."
        )
        df_new = get_stock_data(symbol, start_date=start_date, end_date=end_date)

        # Flatten multi-index columns if necessary
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.get_level_values(0)

        if not df_new.empty:
            df_new.to_parquet(pq_file)
            df_new = format_parquet_columns(df_new, symbol)
            return df_new
        else:
            logger.warning(
                f"No data returned for {symbol} in range {start_ts} - {end_ts}."
            )
            return df_sym

    # If after market close and we have today's data, skip
    if is_after_4pm_est() and pq_file.is_file():
        df_existing = pd.read_parquet(pq_file)
        if not df_existing.empty:
            last_date = df_existing.index.max()
            if last_date is not None and last_date >= today_ts:
                logger.info(
                    f"{symbol}: Already have today's data after market close, skipping."
                )
                return df_existing

    # If before 9:30 AM, shift end_ts to last valid day
    if now_est.time() < datetime.strptime("09:30", "%H:%M").time():
        last_valid_ts = find_last_valid_trading_date(end_ts, tz=est)
        if last_valid_ts < start_ts:
            logger.warning(
                f"{symbol}: No valid trading days in [{start_ts}, {end_ts}]."
            )
            if pq_file.is_file():
                return pd.read_parquet(pq_file)
            return df_sym
        effective_end_ts = min(end_ts, last_valid_ts)
    else:
        effective_end_ts = end_ts

    # Read existing data
    df_sym = pd.read_parquet(pq_file)
    last_date = df_sym.index.max() if not df_sym.empty else None

    # If forced download => redownload everything
    if download:
        logger.info(f"{symbol}: Forced download from {start_ts} to {effective_end_ts}.")
        df_new = get_stock_data(symbol, start_date=start_ts, end_date=effective_end_ts)

        # Flatten multi-index columns if necessary
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.get_level_values(0)

        if not df_new.empty:
            df_new = format_parquet_columns(df_new, symbol=symbol)
            df_sym = pd.concat([df_sym, df_new]).sort_index().drop_duplicates()
            df_sym.to_parquet(pq_file)
        return df_sym

    # If missing data => fetch the missing range
    if last_date is None or last_date < effective_end_ts:
        update_start = (last_date + pd.Timedelta(days=1)) if last_date else start_ts
        logger.info(
            f"{symbol}: Updating data from {update_start} to {effective_end_ts}."
        )
        df_new = get_stock_data(
            symbol, start_date=update_start, end_date=effective_end_ts
        )
        # Flatten multi-index columns if necessary
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.get_level_values(0)

        if not df_new.empty:
            df_new = format_parquet_columns(df_new, symbol=symbol)
            df_sym = pd.concat([df_sym, df_new]).sort_index().drop_duplicates()
            df_sym.to_parquet(pq_file)

    return df_sym


def process_symbols(symbols, start_date, end_date, data_path, download):
    """
    For each symbol:
      1) Load or download data (using yfinance)
      2) Rename 'Adj Close' -> symbol
      3) Join all symbols into a single DataFrame
      4) Return the combined DataFrame
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
        
        # Flatten multi-level columns if present
        if isinstance(df_sym.columns, pd.MultiIndex):
            df_sym.columns = df_sym.columns.get_level_values(0)

        # Rename 'Adj Close' to the ticker symbol, drop the other columns
        try:
            df_sym.rename(columns={"Adj Close": sym}, inplace=True)
            columns_to_drop = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
            for col in columns_to_drop:
                if col in df_sym.columns and col != sym:
                    df_sym.drop(col, axis=1, inplace=True)
        except KeyError as e:
            logger.warning(f"{sym} encountered a key error: {e}")
            continue

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

        # Join the symbol's DataFrame into the main df_all
        if df_all.empty:
            df_all = df_sym
        else:
            df_all = df_all.join(df_sym, how="outer")

    # Fill missing values using forward fill
    df_all.ffill(inplace=True)
    # Fill missing values using backward fill
    df_all.bfill(inplace=True)

    if df_all.isna().any().any():
        logger.warning(
            "Data still has missing values after fill. Dropping remaining nulls."
        )
        df_all.dropna(inplace=True)

    return df_all
