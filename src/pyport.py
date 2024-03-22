"""
Pyport - portfolio optimization
"""
import os
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import csv
import yaml
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

CONFIG_FILENAME = "config.yaml"
with open(CONFIG_FILENAME) as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

if not os.path.exists(config["folder"]):
    os.makedirs(config["folder"])
CWD = os.getcwd() + "/"
PATH = CWD + config["folder"] + "/"

stack, avg, dfs = {}, {}, {}

filtered_times = {k for k in config["models"].keys() if config["models"][k]}
sorted_times = sorted(filtered_times, reverse=True)

for years in sorted_times:
    if not config["models"][years]:
        continue
    
    start_date, end_date = calculate_start_end_dates(years)    
    # get ticker symbols
    if not config["input_files"]:
        pass    
    input_files = config["input_files"] 
    input_files_folder = config.get("input_files_folder", "watchlists")
    input_file_paths = [os.path.join(input_files_folder, file) for file in input_files]
    symbols = process_input_files(input_file_paths)
    
    if config["test_mode"]:
        print(symbols)

    # import data to df
    df = pd.DataFrame()
    needs_update = False

    for sym in symbols:
        if not sym:
            continue
        sym_file = PATH + f"{sym}.csv"        

        if not os.path.exists(sym_file):
            df_sym = get_stock_data(sym, start_date, end_date)
            df_sym.to_csv(sym_file)
        else:
            # print("reading stored data from {}".format(sym_file))            
            df_sym = pd.read_csv(sym_file, parse_dates=True, index_col="Date")
            first_date_index = df_sym.index[0]            
            today = datetime.now().date()
            last_date = get_last_date(sym_file)
            # Check if the first date index is greater than start_date
            if first_date_index > start_date:                
                df_sym = get_stock_data(sym, start_date, today)                
                df_sym.to_csv(sym_file)
            else:                         
                # print("checking {} for new data".format(sym)) 
                if last_date and last_date < today:                    
                    first_valid_date, last_valid_date = get_non_holiday_weekdays(last_date + timedelta(days=1), today)                    
                    if first_valid_date and last_valid_date:                        
                        if last_date < last_valid_date :                            
                            needs_update = True     
                            # print("{} is updating".format(sym))                                    
                            df_sym = update_store(PATH, sym, df_sym, first_valid_date, last_valid_date+ timedelta(days=1))
                  

        df_sym.rename(columns={"Adj Close": sym}, inplace=True)
        df_sym.drop(["Open", "High", "Low", "Close", "Volume"],
                    axis=1,
                    inplace=True)

        # drop by time period        
        date_string_as_timestamp = pd.Timestamp(start_date)       
        differences = abs(df_sym.index - date_string_as_timestamp)

        # Find the index of the minimum timedelta
        closest_index = differences.argmin()

        # Convert the index to a regular column
        df_sym = df_sym.reset_index()

        # Slice df_sym from the closest_index
        df_sym_sliced = df_sym.iloc[closest_index:]
        df_sym = df_sym_sliced.set_index('Date')   

        # append symbol df to main df
        if df.empty:
            df = df_sym
        else:
            df = df.join(df_sym, how="outer")

    df = df.fillna(method='bfill') 
 
    # store df for graphing
    if years == sorted_times[0]:
        dfs["data"] = df
        
        if 'start' in dfs:
            # If dfs["start"] is later than start_date, set it to start_date
            if dfs["start"] > start_date:
                dfs["start"] = start_date    
        else:
            dfs["start"] = start_date    
            
    if "end" in dfs: 
        if dfs["end"] < end_date:
            dfs["end"] = end_date    
    else :
        dfs["end"] = end_date

    if config["test_mode"]:
        df.to_csv("full_df.csv")
        df = df.head(int(len(df) * config["test_data_visible_pct"]))
        print(df.head(), df.tail(), "Null values present: ",
              df.isnull().values.any())

    # for each included model, run optimization
    for optimization in config["models"][years]:
        optimization_method = optimization.lower()

        print(f"\nCalculating {years} {optimization_method.upper()} allocation")

        INPUTS_LIST = ", ".join([str(i) for i in sorted(config["input_files"])])
        model_cache_file = CWD + f"cache/{INPUTS_LIST}-{optimization_method}-{years}.csv"

        if os.path.isfile(model_cache_file):
            with open(model_cache_file, newline="") as cached_data:
                reader = csv.reader(cached_data, delimiter=",")
                result = {row[0]: float(row[1]) for row in reader}

            if holdings_match(cached_model_dict=result, input_file_symbols=symbols, test_mode = config["test_mode"]) and not needs_update:
                modification_time = os.path.getmtime(model_cache_file)
                modification_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modification_time))
                print(f"Model exists in cache/{INPUTS_LIST}-{optimization_method}-{years}.csv. Returning last calculated results from {modification_time_str}")
                weights = pd.DataFrame(result,
                                       index=pd.RangeIndex(start=0,
                                                           stop=1,
                                                           step=1))
                output(
                    data=df,
                    allocation_weights=weights,
                    inputs=INPUTS_LIST,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    stack=stack,
                    sort_by_weights=config["sort_by_weights"],
                    optimization_model=optimization_method,
                    time_period=years,
                    minimum_weight=config["min_weight"],
                    test_mode=config["test_mode"]
                )
                continue

        if config["test_mode"]:
            print("df received by models", df)
        # model handling
        if optimization_method == "hrp":
            temp = HierarchicalRiskParity()
            temp.allocate(
                asset_prices=df,
                linkage=config["optimization_config"][optimization_method]
                ["linkage"],
            )
        elif optimization_method.find("herc") != -1:
            temp = HierarchicalEqualRiskContribution()
            temp.allocate(
                asset_prices=df,
                risk_measure=config["optimization_config"][optimization_method]
                ["risk_measure"],
                linkage=config["optimization_config"][optimization_method]
                ["linkage"],
            )
        elif optimization_method.find("nco") != -1:
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]
            
            temp = NestedClusteredOptimisation()
            if config["optimization_config"][optimization_method]["sharpe"]:
                mu_vec = np.array(asset_returns.mean())
            else:
                mu_vec = np.ones(len(df.columns))
                
            weights = temp.allocate_nco(
                asset_names=df.columns,
                cov=np.array(asset_returns.cov()),
                mu_vec=mu_vec.reshape(-1, 1),
            )
            temp.weights = weights
        elif optimization_method.find("mc") != -1:
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]
            
            temp = NestedClusteredOptimisation()
            mu_vec = np.array(asset_returns.mean())
            w_cvo, w_nco = temp.allocate_mcos(
                mu_vec=mu_vec.reshape(-1, 1),
                cov=np.array(asset_returns.cov()),
                num_obs=config["optimization_config"][optimization_method]
                ["num_obs"],
                num_sims=config["optimization_config"][optimization_method]
                ["num_sims"],
                kde_bwidth=config["optimization_config"][optimization_method]
                ["kde_bandwidth"],
                min_var_portf=not config["optimization_config"]
                [optimization_method]["sharpe"],
                lw_shrinkage=config["optimization_config"][optimization_method]
                ["lw_shrinkage"],
            )
            w_nco = w_nco.mean(axis=0)
            temp_dict = dict(zip(df.columns, w_nco))
            temp.weights = temp_dict
        elif optimization_method.find("cla") != -1:
            temp = CriticalLineAlgorithm()
            solution = config["optimization_config"][optimization_method][
                "solution"]
            temp.allocate(asset_prices=df, solution=solution)
        elif optimization_method == "olmar":
            temp = OLMAR(
                reversion_method=config["optimization_config"]
                [optimization_method]["method"],
                epsilon=config["optimization_config"][optimization_method]
                ["epsilon"],
                window=config["optimization_config"][optimization_method]
                ["window"],
                alpha=config["optimization_config"][optimization_method]
                ["alpha"],
            )
            temp.allocate(asset_prices=df, verbose=config["verbose"])
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif optimization_method == "rmr":
            temp = RMR(
                epsilon=config["optimization_config"][optimization_method]
                ["epsilon"],
                n_iteration=config["optimization_config"][optimization_method]
                ["n_iteration"],
                tau=config["optimization_config"][optimization_method]["tau"],
                window=config["optimization_config"][optimization_method]
                ["window"],
            )
            temp.allocate(asset_prices=df, verbose=config["verbose"])
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif optimization_method == "scorn":
            temp = SCORN(
                window=config["optimization_config"][optimization_method]
                ["window"],
                rho=config["optimization_config"][optimization_method]["rho"],
            )
            temp.allocate(
                asset_prices=df,
                resample_by=config["optimization_config"][optimization_method]
                ["resample"],
                verbose=config["verbose"],
            )
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif optimization_method == "fcornk":
            temp = FCORNK(
                window=config["optimization_config"][optimization_method]
                ["window"],
                rho=config["optimization_config"][optimization_method]["rho"],
                lambd=config["optimization_config"][optimization_method]
                ["lambd"],
                k=config["optimization_config"][optimization_method]["k"],
            )
            temp.allocate(
                asset_prices=df,
                resample_by=config["optimization_config"][optimization_method]
                ["resample"],
                verbose=config["verbose"],
            )
            temp_dict = dict(zip(df.columns, temp.weights[0]))
            temp.weights = temp_dict
        else:
            temp = MeanVarianceOptimisation()
            expected_returns = ReturnsEstimators(
            ).calculate_mean_historical_returns(asset_prices=df)
            covariance = ReturnsEstimators().calculate_returns(
                asset_prices=df).cov()
            temp.allocate(
                asset_names=df.columns,
                asset_prices=df,
                expected_asset_returns=expected_returns,
                covariance_matrix=covariance,
                solution=optimization_method,
                target_return=config["optimization_config"]["efficient_risk"],
                target_risk=config["optimization_config"]["efficient_return"],
                risk_aversion=config["optimization_config"]["risk_aversion"],
            )
            temp.get_portfolio_metrics()

        # send to output
        output(
            data=df,
            allocation_weights=temp.weights,
            inputs=INPUTS_LIST,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            stack=stack,
            sort_by_weights=config["sort_by_weights"],
            optimization_model=optimization_method,
            time_period=years,
            minimum_weight=config["min_weight"],
        )

if len(stack) > 0:
    avg = stacked_output(stack)
    sorted_avg = dict(sorted(avg.items(), key=lambda item: item[1]))
    min_weight = get_min_by_size(sorted_avg, config["portfolio_max_size"])
    models = {k: v for k, v in config["models"].items() if v is not None}

    daily_returns_to_plot, cumulative_returns_to_plot = output(
        data=dfs["data"],
        allocation_weights=sorted_avg,
        inputs=", ".join([str(i) for i in sorted(config["input_files"])]),
        sort_by_weights=True,
        start_date=dfs["start"],
        end_date=dfs["end"],
        stack=stack,
        symbols=symbols,
        optimization_model=", ".join(sorted(list(set(sum(models.values(),
                                                         []))))),
        time_period=sorted_times[0],
        minimum_weight=min_weight,
        max_size=config["portfolio_max_size"],
    )
    
    plot_graphs(daily_returns_to_plot, cumulative_returns_to_plot, avg, config)

if __name__ == "__main__":
    cache_dir = "cache"
    cleanup_cache(cache_dir)