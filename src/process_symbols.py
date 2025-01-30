from datetime import datetime
from pathlib import Path
import pandas as pd
import pytz

from utils import logger
from utils.data_utils import flatten_columns, format_to_df_format, get_stock_data
from utils.date_utils import find_valid_trading_date, is_after_4pm_est


def load_or_download_symbol_data(symbol, start_date, end_date, data_path, download):
    """
    Load symbol data from a Parquet file or download if missing or outdated.
    Ensures stock data is correctly aligned to available history.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    pq_file = Path(data_path) / f"{symbol}.parquet"

    est = pytz.timezone("US/Eastern")
    now_est = datetime.now(est)
    today_ts = pd.Timestamp(now_est.date())

    # 1) If file doesn't exist, download full range
    if not pq_file.is_file():
        logger.info(
            f"No file for {symbol}. Downloading full history from {start_ts} to {end_ts}."
        )
        df_new = get_stock_data(symbol, start_date=start_ts, end_date=end_ts)
        df_new = flatten_columns(df_new, symbol)

        if df_new.empty:
            logger.warning(f"No data for {symbol} in range {start_ts} - {end_ts}.")
            return pd.DataFrame()

        df_new.to_parquet(pq_file)
        return format_to_df_format(df_new, symbol)

    # 2) If after market close and we have today's data, skip
    if is_after_4pm_est() and pq_file.is_file():
        df_existing = pd.read_parquet(pq_file)
        df_existing = flatten_columns(df_existing, symbol)
        last_date = df_existing.index.max()
        if last_date is not None and last_date >= today_ts:
            logger.info(
                f"{symbol}: Already have today's data after market close, skipping."
            )
            return format_to_df_format(df_existing, symbol)

    # 3) Adjust end_ts if before market open
    effective_end_ts = end_ts
    if now_est.time() < datetime.strptime("09:30", "%H:%M").time():
        effective_end_ts = find_valid_trading_date(end_ts, tz=est, direction="backward")

    # Ensure we only update up to the last valid trading day before today
    if effective_end_ts.normalize() >= today_ts.normalize():
        effective_end_ts = find_valid_trading_date(
            today_ts - pd.Timedelta(days=1), tz=est, direction="backward"
        )

    # 4) Read existing data
    df_existing = pd.read_parquet(pq_file)
    df_existing = flatten_columns(df_existing, symbol)
    last_date = df_existing.index.max() if not df_existing.empty else None

    # 5) Handle forced download
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
        return format_to_df_format(df_existing, symbol)

    # 6) Append missing data dynamically
    if last_date is None or last_date < effective_end_ts:
        update_start = last_date + pd.Timedelta(days=1) if last_date else start_ts
        update_start = find_valid_trading_date(
            update_start, tz=est, direction="forward"
        )

        if update_start >= effective_end_ts:
            logger.debug(f"{symbol}: Data already up-to-date.")
            return format_to_df_format(df_existing, symbol)

        logger.info(f"{symbol}: Updating from {update_start} to {effective_end_ts}.")
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
            logger.info(f"{symbol}: No new data found for update period.")
            return format_to_df_format(df_existing, symbol)
    else:
        return format_to_df_format(df_existing, symbol)


def process_symbols(symbols, start_date, end_date, data_path, download):
    """
    Loads or downloads data for all symbols, ensuring proper historical alignment.
    """
    df_all = pd.DataFrame()

    for sym in symbols:
        df_sym = load_or_download_symbol_data(
            sym, start_date, end_date, data_path, download
        )
        if df_sym.empty:
            logger.warning(f"No data for {sym}, skipping.")
            continue

        df_sym = df_sym.loc[~df_sym.index.duplicated(keep="first")]
        df_sym = format_to_df_format(df_sym, sym)

        # Ensure we respect each stock's available data instead of slicing arbitrarily
        earliest = df_sym.index.min()
        logger.info(f"{sym} starts at {earliest}. Using full available history.")

        # Convert columns to multi-level (Outer = ticker, Inner = OHLC data)
        df_sym.columns = pd.MultiIndex.from_product([[sym], df_sym.columns])

        # Merge into main DataFrame
        df_all = df_sym if df_all.empty else df_all.join(df_sym, how="outer")

    # Fill missing values forward, then backward
    df_all.ffill(inplace=True)
    df_all.bfill(inplace=True)

    # Drop any remaining nulls
    if df_all.isna().any().any():
        logger.warning(
            "Data still has missing values after fill. Dropping remaining nulls."
        )
        df_all.dropna(inplace=True)

    return df_all
