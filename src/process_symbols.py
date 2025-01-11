

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pytz

from utils import logger
from utils.data_utils import get_stock_data, get_last_date, update_store
from utils.date_utils import get_non_holiday_weekdays, is_after_4pm_est, is_weekday


def load_or_download_symbol_data(symbol, start_date, end_date, data_path, download):
    """
    Load the symbol data from a Parquet file, or download it if the file doesn't exist
    or needs an update. Append missing data to the Parquet file if necessary.

    We'll skip downloading on weekends, holidays, and after market close
    if data for today is already present.
    """
    pq_file = Path(data_path) / f"{symbol}.parquet"
    df_sym = pd.DataFrame()  

    est = pytz.timezone("US/Eastern")
    now_est = datetime.now(est)
    today = now_est.date()

    # 1) If not a weekday, just return what we have (or empty if no file)
    if not is_weekday(today):
        return pd.read_parquet(pq_file) if pq_file.exists() else df_sym

    # 2) If today is a holiday on the NYSE, skip
    first_valid_date, _ = get_non_holiday_weekdays(today, today, tz=est)
    if today != first_valid_date:
        return pd.read_parquet(pq_file) if pq_file.exists() else df_sym

    # 3) If it's after 4:01 PM EST, check if we already have today's data
    after_market_close = is_after_4pm_est()
    if after_market_close and pq_file.is_file():
        df_sym = pd.read_parquet(pq_file)
        last_date = get_last_date(pq_file)  

        # If we already have today's data, no need to re-download
        if last_date is not None and last_date >= pd.Timestamp(today):
            return df_sym

    # 4) If the file exists and we're NOT forcing a download, maybe append missing data
    if pq_file.is_file() and not download:
        df_sym = pd.read_parquet(pq_file)

        first_date = df_sym.index[0] if not df_sym.empty else None
        last_date = get_last_date(pq_file)

        # If the first date is more recent than start_date, append missing older data
        if first_date and first_date > pd.Timestamp(start_date):
            logger.info(f"Appending missing data from {start_date} to {first_date - pd.Timedelta(days=1)} for {symbol}")
            missing_data = get_stock_data(symbol, start_date, first_date - pd.Timedelta(days=1))
            df_sym = pd.concat([missing_data, df_sym]).sort_index().drop_duplicates()
            df_sym.to_parquet(pq_file)

        # If last_date is stale, update from last_date+1 day to end_date
        if last_date and last_date < end_date:
            logger.info(f"Updating {symbol} data from {last_date+pd.Timedelta(days=1)} to {end_date}")
            df_sym = update_store(data_path, symbol, last_date + timedelta(days=1), end_date)
    else:
        # 5) Otherwise, either the file doesn't exist or we forced a re-download
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        df_sym = get_stock_data(symbol, start_date=start_date, end_date=end_date)
        if not df_sym.empty:
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
        if not sym:
            continue

        # Load or download the data
        df_sym = load_or_download_symbol_data(sym, start_date, end_date, data_path, download)

        # Drop duplicates in the index (if any)
        df_sym = df_sym.loc[~df_sym.index.duplicated(keep="first")]

        # Rename 'Adj Close' to the ticker symbol
        # and drop the other columns (Open, High, Low, Close, Volume, etc.)
        try:
            df_sym.rename(columns={"Adj Close": sym}, inplace=True)
            columns_to_drop = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
            for col in columns_to_drop:
                if col in df_sym.columns and col != sym:
                    df_sym.drop(col, axis=1, inplace=True)
        except KeyError as e:
            logger.warning(f"{sym} encountered a key error: {e}")

        # Slice from the start_date onward
        try:
            pos = df_sym.index.get_loc(start_date_ts, method="nearest")
            df_sym = df_sym.iloc[pos:]
        except KeyError:
            logger.warning(f"KeyError: {start_date_ts} not found in {sym}'s data index.")
            continue

        # Join into the main DataFrame
        if df_all.empty:
            df_all = df_sym
        else:
            df_all = df_all.join(df_sym, how="outer")

    # Fill leading NaNs with the next valid data
    df_all.fillna(method="bfill", inplace=True)

    return df_all
