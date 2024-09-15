"""
Pyport - portfolio optimization
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import logging
from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution
from portfoliolab.clustering.hrp import HierarchicalRiskParity
from portfoliolab.modern_portfolio_theory.mean_variance import MeanVarianceOptimisation
from portfoliolab.modern_portfolio_theory.mean_variance import ReturnsEstimators
from portfoliolab.modern_portfolio_theory import CriticalLineAlgorithm
from portfoliolab.clustering.nco import NestedClusteredOptimisation
from portfoliolab.online_portfolio_selection.rmr import RMR
from portfoliolab.online_portfolio_selection.olmar import OLMAR
from portfoliolab.online_portfolio_selection.fcornk import FCORNK
from portfoliolab.online_portfolio_selection.scorn import SCORN
from date_helpers import *
from plotly_graphs import plot_graphs
from stock_download import *
from portfolio import *
from output import *
from helpers import *
from caching import *

# Setup logger
logging.basicConfig(
    level=logging.INFO, 
    format='\n%(levelname)s: %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
class Config:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.folder = self.config["folder"]
        self.input_files_folder = self.config.get("input_files_folder", "watchlists")
        self.input_files = self.config["input_files"]
        self.models = self.config["models"]
        self.download = self.config.get("download", False)
        self.plot_daily_returns = self.config.get("plot_daily_returns", False)
        self.plot_cumulative_returns = self.config.get("plot_cumulative_returns", False)
        self.min_weight = self.config["min_weight"]
        self.portfolio_max_size = self.config["portfolio_max_size"]        
        self.sort_by_weights = self.config.get("sort_by_weights", False)
        self.verbose = self.config.get("verbose", False)
        self.test_mode = self.config.get("test_mode", False)
        self.test_data_visible_pct = self.config["test_data_visible_pct"]
        self.optimization_config = self.config["optimization_config"]
        
def load_or_download_symbol_data(symbol, start_date, end_date, data_path, download):
    """
    Load the symbol data from a CSV file, or download it if the file doesn't exist
    or needs an update. Append missing data to the CSV from the selected start_date 
    if the file starts later than the start_date. Skip downloading on weekends, holidays,
    and after market close if data for today has already been downloaded.
    """
    symbol_file = Path(data_path) / f"{symbol}.csv"
    
    # Initialize an empty dataframe for the symbol
    df_sym = pd.DataFrame()

    # Get the current time in EST
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    
    # Determine if today is a weekend
    today = now_est.date()
    if not is_weekday(today):    
        return pd.read_csv(symbol_file, parse_dates=True, index_col="Date") if symbol_file.exists() else df_sym
    
    # Check if today is a valid trading day (not a holiday)
    first_valid_date, _ = get_non_holiday_weekdays(today, today, tz=est)
    if today != first_valid_date:       
        return pd.read_csv(symbol_file, parse_dates=True, index_col="Date") if symbol_file.exists() else df_sym

    # Determine if it's after 4:01 PM EST
    after_market_close = is_after_4pm_est()

    # If it's after market close and data for today exists, skip the download
    if after_market_close and symbol_file.exists():
        df_sym = pd.read_csv(symbol_file, parse_dates=True, index_col="Date")
        last_date = get_last_date(symbol_file)  # Get the last date from the file

        # Check if the data for today has already been downloaded
        if last_date is not None and last_date >= pd.Timestamp(today):         
            return df_sym

    # If the file exists and downloading is not forced, update the file if necessary
    if symbol_file.exists() and not download:
        df_sym = pd.read_csv(symbol_file, parse_dates=True, index_col="Date")
        first_date = df_sym.index[0]  # First row date
        last_date = get_last_date(symbol_file)  # Last row date

        # If the first row's date is later than start_date, download missing data and append it
        if first_date > pd.Timestamp(start_date):
            print(f"Appending missing data from {start_date} to {first_date} for {symbol}")
            missing_data = get_stock_data(symbol, start_date=start_date, end_date=first_date - pd.Timedelta(days=1))
            # Append the missing data at the top
            df_sym = pd.concat([missing_data, df_sym])

            # Save the updated dataframe to the file
            df_sym.to_csv(symbol_file)

        # Update the file with data from last_date to end_date if the data is outdated
        if last_date is not None and last_date < end_date:
            print(f"Updating {symbol} data from {last_date} to {end_date}")
            df_sym = update_store(data_path, symbol, df_sym, last_date, end_date)
    else:
        # Download and save the data if the file doesn't exist or download is forced
        print(f"Downloading {symbol} data from {start_date} to {end_date}")
        df_sym = get_stock_data(symbol, start_date=start_date, end_date=end_date)
        df_sym.to_csv(symbol_file)

    return df_sym


def process_symbols(symbols, start_date, end_date, path, download):
    df = pd.DataFrame()

    # Convert start_date to pandas.Timestamp to ensure it matches the index type
    start_date_ts = pd.Timestamp(start_date)

    for sym in symbols:
        if not sym:
            continue

        df_sym = load_or_download_symbol_data(sym, start_date, end_date, path, download)

        # Ensure the index is unique by dropping duplicate dates
        df_sym = df_sym[~df_sym.index.duplicated(keep='first')]

        df_sym.rename(columns={"Adj Close": sym}, inplace=True)
        df_sym.drop(["Open", "High", "Low", "Close", "Volume"], axis=1, inplace=True)

        try:
            # Get the positional index of the closest date
            pos = df_sym.index.get_loc(start_date_ts, method='nearest')

            # Use .iloc to slice by position
            df_sym = df_sym.iloc[pos:]
        except KeyError:
            print(f"KeyError: {start_date_ts} not found in {sym}'s data")
            continue

        if df.empty:
            df = df_sym
        else:
            df = df.join(df_sym, how="outer")

    return df.fillna(method='bfill')

def run_optimization(method, df, config):
    OPTIMIZATION_METHODS = {
        "hrp": HierarchicalRiskParity,
        "herc": HierarchicalEqualRiskContribution,
        "nco": NestedClusteredOptimisation,
        "mc": NestedClusteredOptimisation,
        "mc2": NestedClusteredOptimisation,
        "cla": CriticalLineAlgorithm,
        "cla2": CriticalLineAlgorithm,
        "olmar": OLMAR,
        "rmr": RMR,
        "scorn": SCORN,
        "fcornk": FCORNK,
        "mean_variance": MeanVarianceOptimisation,
    }
    
    optimizer_class = OPTIMIZATION_METHODS.get(method)
    if optimizer_class:
        if method == 'olmar':
            optimizer = optimizer_class(
                reversion_method=config[method]["method"], 
                epsilon=config[method]["epsilon"],
                window=config[method]["window"],
                alpha=config[method]["alpha"],
            )
        elif method == 'rmr':
            optimizer = optimizer_class(
                epsilon=config[method]["epsilon"],
                n_iteration=config[method]["n_iteration"],
                tau=config[method]["tau"],
                window=config[method]["window"],
            )
        elif method == 'scorn':
            optimizer = optimizer_class(
                window=config[method]["window"],
                rho=config[method]["rho"],
            )
        elif method == 'fcornk':
            optimizer = optimizer_class(
                window=config[method]["window"],
                rho=config[method]["rho"],
                lambd=config[method]["lambd"],
                k=config[method]["k"],
            )
        else :
            optimizer = optimizer_class()

        # Handle NCO with Sharpe check
        if method == "nco":
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]
            
            if config[method]["sharpe"]:
                mu_vec = np.array(asset_returns.mean())
            else:
                mu_vec = np.ones(len(df.columns))

            weights = optimizer.allocate_nco(
                asset_names=df.columns,
                cov=np.array(asset_returns.cov()),
                mu_vec=mu_vec.reshape(-1, 1),
            )
            optimizer.weights = weights
        
        # Handle MC optimization
        elif method in ["mc", "mc2"]:
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]
            mu_vec = np.array(asset_returns.mean())

            w_cvo, w_nco = optimizer.allocate_mcos(
                mu_vec=mu_vec.reshape(-1, 1),
                cov=np.array(asset_returns.cov()),
                num_obs=config[method]["num_obs"],
                num_sims=config[method]["num_sims"],
                kde_bwidth=config[method]["kde_bandwidth"],
                min_var_portf=not config[method]["sharpe"],
                lw_shrinkage=config[method]["lw_shrinkage"],
            )
            w_nco = w_nco.mean(axis=0)
            optimizer.weights = dict(zip(df.columns, w_nco))

        # Handle Critical Line Algorithm (CLA)
        elif method in ["cla", "cla2"]:
            solution = config[method]["solution"]
            optimizer.allocate(asset_prices=df, solution=solution)
        
        # Handle Mean-Variance Optimization
        elif method == "mean_variance":
            expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=df)
            covariance = ReturnsEstimators().calculate_returns(asset_prices=df).cov()

            optimizer.allocate(
                asset_names=df.columns,
                asset_prices=df,
                expected_asset_returns=expected_returns,
                covariance_matrix=covariance,
                solution=method,
                target_return=config["efficient_risk"],
                target_risk=config["efficient_return"],
                risk_aversion=config["risk_aversion"],
            )
            optimizer.get_portfolio_metrics()

        elif method in ["fcornk", "scorn", "rmr", "olmar"]:
            # Safely get the 'resample' parameter, defaulting to None if not present
            resample_by = config[method].get("resample", None)
            
            # Call allocate with or without resample based on the configuration
            if resample_by:
                optimizer.allocate(asset_prices=df, resample_by=resample_by)
            else:
                optimizer.allocate(asset_prices=df)
        # For all other optimizations
        else:
            optimizer.allocate(asset_prices=df, **config[method])

        return optimizer
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def run_optimization_and_save(df, config, start_date, end_date, symbols, stack, years):
    """Runs the optimization process, checks cache, saves results in cache, and skips recalculation if valid cache exists."""
    for optimization in config.models[years]:
        optimization_method = optimization.lower()
        model_name = optimization_method + str(years)
        input_filename = ", ".join([str(i) for i in sorted(config.input_files)])  # Combined input file names
        
        logger.info(f"\nCalculating {years} {optimization_method.upper()} allocation")
        
        # Check if results already exist in cache
        cached_results = load_model_results_from_cache(model_name, years, input_filename)
        
        # If cached results exist, use them
        if cached_results:
            print(f"Using cached results for {years} {optimization_method.upper()} allocation")
            normalized_weights = normalize_weights(cached_results, config.config["min_weight"])
            stack[model_name] = normalized_weights
        else:
            # Cache does not exist, proceed with optimization
            try:
                # Run optimization
                optimizer = run_optimization(optimization_method, df, config.config["optimization_config"])
                
                # Convert optimizer.weights to dict (assuming asset_names are in symbols)
                converted_weights = convert_to_dict(optimizer.weights, asset_names=symbols)
                
                # Normalize weights before adding to stack
                normalized_weights = normalize_weights(converted_weights, config.config["min_weight"])
                
                # Add to stack (done here instead of in `output`)
                stack[model_name] = normalized_weights
                
                # Save the results to the cache
                save_model_results(model_name, years, input_filename, symbols, normalized_weights)
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                continue  # Skip to the next iteration on error
        
        # Output results (whether from cache or recalculated)
        output_results(df, normalized_weights, config, start_date, end_date, years)

    
def main():
    CONFIG_FILENAME = "config.yaml"
    config = Config(CONFIG_FILENAME)

    # Setup directories
    CWD = Path.cwd()
    PATH = CWD / config.folder
    PATH.mkdir(parents=True, exist_ok=True)

    stack, avg, dfs = {}, {}, {}

    # Filter time periods
    filtered_times = [k for k in config.models.keys() if config.models[k]]
    sorted_times = sorted(filtered_times, reverse=True)

     # Process symbols and load data
    for years in sorted_times:
        start_date, end_date = calculate_start_end_dates(years)
        symbols = process_input_files([Path(config.input_files_folder) / file for file in config.input_files])
        
        if config.test_mode:
            logger.info(f"Test mode: symbols - {symbols}")

        df = process_symbols(symbols, start_date, end_date, PATH, config.download)

        # Store results in dfs
        if years == sorted_times[0]:
            dfs["data"] = df
            dfs["start"] = start_date
            dfs["end"] = end_date
        else:
            dfs["start"] = min(dfs.get("start", start_date), start_date)
            dfs["end"] = max(dfs.get("end", end_date), end_date)

        # Store full dataframe for test mode
        if config.test_mode:
            df.to_csv("full_df.csv")
            df = df.head(int(len(df) * config.config["test_data_visible_pct"]))

        run_optimization_and_save(df, config, start_date, end_date, symbols, stack, years)

    # Post-processing: Handle stack and averaging results    
    if len(stack) > 0:        
        avg = stacked_output(stack)
        sorted_avg = dict(sorted(avg.items(), key=lambda item: item[1]))
        normalized_avg = normalize_weights(sorted_avg, config.config["min_weight"])
    
        # Filter out None values from config.models
        valid_models = [v for v in config.models.values() if v is not None]    
        # Combine model names for the final output    
        combined_model_names = ", ".join(sorted(list(set(sum(valid_models, [])))))

        combined_input_files_names = ", ".join([str(i) for i in sorted(config.input_files)])

        # Plot final graphs using the averaged results
        daily_returns_to_plot, cumulative_returns_to_plot = output(
            data=dfs["data"],
            allocation_weights=normalized_avg,
            inputs=combined_input_files_names,
            start_date=dfs["start"],
            end_date=dfs["end"],
            optimization_model=combined_model_names,
            time_period=sorted_times[0],
            minimum_weight=config.config["min_weight"],
            max_size=config.config["portfolio_max_size"],
            config=config
        )

        # Plot graphs
        plot_graphs(daily_returns_to_plot, cumulative_returns_to_plot, avg, config, symbols)

    if __name__ == "__main__":
        cache_dir = "cache"
        cleanup_cache(cache_dir)
    
    # Run the script
if __name__ == "__main__":
    main()