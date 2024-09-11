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
from stock_download import *
from portfolio import *
from output import *

# Setup logger
logging.basicConfig(level=logging.INFO)
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
    or needs an update. Append missing data to the CSV.
    """
    symbol_file = Path(data_path) / f"{symbol}.csv"
    
    # Initialize an empty dataframe for the symbol
    df_sym = pd.DataFrame()

    if symbol_file.exists() and not download:
        # Read existing data
        df_sym = pd.read_csv(symbol_file, parse_dates=True, index_col="Date")
        last_date = get_last_date(symbol_file)
        
        if last_date is not None and last_date < end_date:
            # Update the file with data from last_date to end_date
            print(f"Updating {symbol} data from {last_date} to {end_date}")
            df_sym = update_store(data_path, symbol, df_sym, last_date, end_date)
    else:
        # Download and save the data if file doesn't exist or download is forced
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
                reversion_method=config["optimization_config"]
                    [method]["method"],
                    epsilon=config["optimization_config"][method]
                    ["epsilon"],
                    window=config["optimization_config"][method]
                    ["window"],
                    alpha=config["optimization_config"][method]
                    ["alpha"],
            )
        elif method == 'rmr':
            optimizer = optimizer_class(
                 epsilon=config["optimization_config"][method]
                    ["epsilon"],
                    n_iteration=config["optimization_config"][method]
                    ["n_iteration"],
                    tau=config["optimization_config"][method]["tau"],
                    window=config["optimization_config"][method]
                    ["window"],
            )
        elif method == 'scorn':
            optimizer = optimizer_class(
                  window=config["optimization_config"][method]
                    ["window"],
                    rho=config["optimization_config"][method]["rho"],
            )
        elif method == 'fcornk':
            optimizer = optimizer_class(
                 window=config["optimization_config"][method]
                    ["window"],
                    rho=config["optimization_config"][method]["rho"],
                    lambd=config["optimization_config"][method]
                    ["lambd"],
                    k=config["optimization_config"][method]["k"],
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

        # For all other optimizations
        else:
            optimizer.allocate(asset_prices=df, **config[method])

        return optimizer
    else:
        raise ValueError(f"Unknown optimization method: {method}")

def output_results(df, weights, config, start_date, end_date, symbols, stack, years):
    output(
        data=df,
        allocation_weights=weights,
        inputs=", ".join([str(i) for i in sorted(config.input_files)]),
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        stack=stack,
        optimization_model=", ".join(sorted(list(set(sum(config.models.values(), []))))),
        time_period=years,
        minimum_weight=config.config["min_weight"],
        max_size=config.config.get("portfolio_max_size", 0),
        config=config
    )
    
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

        for optimization in config.models[years]:
            optimization_method = optimization.lower()
            logger.info(f"\nCalculating {years} {optimization_method.upper()} allocation")
            
            try:
                optimizer = run_optimization(optimization_method, df, config.config["optimization_config"])
                print(optimization_method, optimizer.weights)
                # Store the result in stack
                stack[years] = optimizer.weights                
                # Output results
                output_results(df, optimizer.weights, config, start_date, end_date, symbols, stack, years)

            except Exception as e:
                logger.error(f"Optimization error: {e}")

    # Post-processing: Handle stack and averaging results
    if len(stack) > 0:
        avg = stacked_output(stack)
        sorted_avg = dict(sorted(avg.items(), key=lambda item: item[1]))
        min_weight = get_min_by_size(sorted_avg, config.config["portfolio_max_size"])

        # Plot final graphs using the averaged results
        daily_returns_to_plot, cumulative_returns_to_plot = output(
            data=dfs["data"],
            allocation_weights=sorted_avg,
            inputs=", ".join([str(i) for i in sorted(config.input_files)]),
            start_date=dfs["start"],
            end_date=dfs["end"],
            stack=stack,
            symbols=symbols,
            optimization_model=", ".join(sorted(list(set(sum(config.models.values(), []))))),
            time_period=sorted_times[0],
            minimum_weight=min_weight,
            max_size=config.config["portfolio_max_size"],
            config=config
        )

        # Plot graphs
        plot_graphs(daily_returns_to_plot, cumulative_returns_to_plot, avg, config)

    if __name__ == "__main__":
        cache_dir = "cache"
        cleanup_cache(cache_dir)
    
    # Run the script
if __name__ == "__main__":
    main()