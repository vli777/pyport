"""
Pyport - portfolio optimization
"""
import os
import re
import csv
from datetime import datetime, timedelta
from collections import Counter
import sys
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import yfinance as yf
import yaml
from playsound import playsound
from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution
from portfoliolab.clustering.hrp import HierarchicalRiskParity
from portfoliolab.modern_portfolio_theory.mean_variance import MeanVarianceOptimisation
from portfoliolab.modern_portfolio_theory.mean_variance import ReturnsEstimators
from portfoliolab.modern_portfolio_theory import CriticalLineAlgorithm
from portfoliolab.clustering.nco import NestedClusteredOptimisation
# from portfoliolab.estimators.risk_estimators import RiskEstimators
from portfoliolab.online_portfolio_selection.rmr import RMR
from portfoliolab.online_portfolio_selection.olmar import OLMAR
from portfoliolab.online_portfolio_selection.fcornk import FCORNK
from portfoliolab.online_portfolio_selection.scorn import SCORN
from mlfinlab.backtest_statistics import sharpe_ratio, drawdown_and_time_under_water
from pandas.tseries.holiday import USFederalHolidayCalendar

# setup
CONFIG_FILENAME = "config.yaml"
with open(CONFIG_FILENAME) as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

## global vars
stk, avg, dfs = {}, {}, {}
TODAY = datetime.today()


if not os.path.exists(config["folder"]):
    os.makedirs(config["folder"])
CWD = os.getcwd() + "/"
PATH = CWD + config["folder"] + "/"

def cleanup_cache(cache_dir, max_age_hours):
    now = datetime.now()
    max_age = timedelta(hours=max_age_hours)

    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
            if now - creation_time > max_age:
                os.remove(filepath)
                print(f"Deleted old file: {filename}")

def str_to_date(date_str, fmt="%Y-%m-%d"):
    """
    convert string to datetime
    """
    return datetime.strptime(date_str, fmt)


def date_to_str(date, fmt="%Y-%m-%d"):
    """
    convert datetime to string
    """
    return date.strftime(fmt)


def get_stock_data(symbol, start_date, end_date):
    """
    download stock price data from yahoo finance with optional write to csv
    """
    print(f"Downloading {symbol} {start_date} - {end_date} ...")
    symbol_df = yf.download(symbol, start=start_date, end=end_date)
    return symbol_df

def update_store(symbol, symbol_df, start_date, end_date):
    new_data = get_stock_data(symbol, start_date=start_date, end_date=end_date)
    
    # Append new data to the CSV file
    csv_filename = PATH + f"{symbol}.csv"
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write each row with the index included
        for index, row in new_data.iterrows():
            index_date_string = index.strftime('%Y-%m-%d')  # Convert index to date string
            writer.writerow([index_date_string] + row.tolist())

    # Update symbol_df with new data
    symbol_df = pd.concat([symbol_df, new_data], axis=0)    
    return symbol_df

def scale_to_one(weights_dict):
    """
    scaling function to have filtered holding allocations sum to one
    """
    total_alloc = sum(weights_dict.values())
    scaled = {k: v / total_alloc for k, v in weights_dict.items()}
    return scaled


def custom_scaling(weights_dict, scaling):
    """
    returns a weights dict with custom scaled values to inc/dec the impact of an allocation result
    """
    return {k: v * scaling for k, v in weights_dict.items()}


def stacked_output(stack_dict):
    """
    return a scaled arithmetic avg of input model dicts
    """
    maxlen = max([v[1] for v in stack_dict.values()])

    for model in stack_dict:
        portfolio, scaling_factor = stack_dict[model]
        stack_dict[model] = custom_scaling(weights_dict=portfolio,
                                           scaling=scaling_factor / maxlen)

    holding = [v for _, v in stack_dict.items()]
    total = sum(map(Counter, holding), Counter())
    average_holdings = {k: v / len(stack_dict) for k, v in total.items()}
    return average_holdings


def apply_weights(row, weights_dict):
    """
    apply scaling weights from a custom dict to a row
    """
    for i, _ in enumerate(row):
        row[i] *= weights_dict[i]
    return row


def clip_by_weight(weights_dict, mininum_weight):
    """
    filters any holdings below a min threshold
    """
    return {k: v for k, v in weights_dict.items() if v > mininum_weight}


def get_min_by_size(weights_dict, size, mininum_weight=0.01):
    """
    find the minimum allocation of a weight dict
    """
    if len(weights_dict) > size:
        sorted_weights = sorted(weights_dict.values(), reverse=True)
        mininum_weight = sorted_weights[size]
    return mininum_weight


def holdings_match(cached_dict, symbol_list):
    """
    check if all selected symbols match between the input list and cache files
    """
    for symbol in cached_dict.keys():
        if symbol not in symbol_list:
            if config["test_mode"]:
                print(symbol, "not found in", symbols)
            return False
    for symbol in symbol_list:
        if symbol not in cached_dict.keys():
            if config["test_mode"]:
                print(symbol, "not found in", cached_dict.keys())
            return False
    return True


def output(
    data,
    allocation_weights,
    inputs,
    start_date,
    end_date,
    max_size=21,
    sort_by_weights=False,
    optimization_model=None,
    time_period=1.0,
    minimum_weight=0.01,
):
    """
    produces console output with descriptive statistics of the provided portfolio over a date range
    """
    if isinstance(allocation_weights, dict):
        clean_weights = allocation_weights
    else:
        clean_weights = allocation_weights.to_dict("records")[0]
    if config["test_mode"]:
        print("raw weights", clean_weights)

    scaled = scale_to_one(clean_weights)

    # if (len(scaled) == 1 or any(np.isnan(val) for val in scaled.values()) or
    #         max(scaled.values()) < minimum_weight):
    #     scaled = {"SPY": 1}
    while len(scaled) > 0 and min(scaled.values()) < minimum_weight or len(scaled) > max_size:
        clipped = clip_by_weight(scaled, minimum_weight)
        scaled = scale_to_one(clipped)
        minimum_weight = get_min_by_size(scaled, max_size, minimum_weight)

    # store portfolio
    stk[optimization_model + str(time_period)] = [scaled, len(scaled)]
    if not os.path.exists(CWD + "cache"):
        os.makedirs(CWD + "cache")
    output_file = (
        CWD + "cache/" + f"{inputs}-{optimization_model}-{time_period}.csv")

    writer = csv.writer(open(output_file, "w", newline=""))
    for key, val in scaled.items():
        writer.writerow([key, val])
    filtered_symbols = [sym for sym in symbols if sym not in scaled.keys()]
    for symbol in filtered_symbols:
        writer.writerow([symbol, 0])

    if len(scaled) > 0:
        portfolio = data[scaled.keys()]
        returns = np.log(portfolio) - np.log(portfolio.shift(1))
        returns = returns.iloc[1:, :]
        weights_vector = list(scaled.values())

        weighted_returns = returns.mul(weights_vector)   
        cumulative_returns = weighted_returns.add(1).cumprod() 

        portfolio_returns = weighted_returns.apply(np.sum, axis=1)      
        portfolio_cumulative_returns = portfolio_returns.add(1).cumprod()        

        try:
            sharpe = sharpe_ratio(portfolio_returns)    
        except:
            sharpe = 0

        print(f"\ntime period: {start_date} to {end_date} ({time_period} yrs)")
        print("inputs:", inputs)
        print("optimization methods:", optimization_model)
        print("sharpe ratio:", round(sharpe, 2))
        print(f"cumulative return: { round((portfolio_cumulative_returns[-1] - 1) * 100, 2)}%")
        print(
            f"portfolio allocation weights (min {minimum_weight:.2f}):")
    else:
        print("max diversification recommended")

    if sort_by_weights:
        for symbol, weight in sorted(scaled.items(),
                                     key=lambda kv: (kv[1], kv[0]),
                                     reverse=True):
            print(symbol, f"\t{weight:.3f}")
    else:
        for symbol, weight in sorted(scaled.items()):
            print(symbol, f"\t{weight:.3f}")

    portfolio_returns = portfolio_returns.to_frame()
    portfolio_returns.columns=['SIM_PORT']

    portfolio_cumulative_returns = portfolio_cumulative_returns.to_frame()
    portfolio_cumulative_returns.columns=['SIM_PORT']

    all_daily_returns = returns.join(portfolio_returns)
    all_cumulative_returns = ((portfolio_cumulative_returns) - 1).join(returns.add(1).cumprod() - 1)    
    return all_daily_returns, all_cumulative_returns

def plot_graphs(daily_returns, cumulative_returns):
    """
    creates plotly graphs
    """
    if config["test_mode"]:
        print(daily_returns, cumulative_returns)
    if config["plot_cumulative_returns"] or config["plot_daily_returns"]:
        # daily_returns = data.pct_change()[1:]
        if config["test_mode"]:
            print(daily_returns.head())

        # cumulative_returns = daily_returns.add(1).cumprod().sub(1).mul(100)
        if config["test_mode"]:
            print(cumulative_returns.head())

        if config["plot_cumulative_returns"]:
            if config["sort_by_weights"]:
                sorted_cols = cumulative_returns.sort_values(
                    cumulative_returns.index[-1], ascending=False,
                    axis=1).columns
                cumulative_returns = cumulative_returns[sorted_cols]

                fig = go.Figure()

                for col in cumulative_returns.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns[col],
                            mode="lines" if col != "SIM_PORT" else "lines+markers",
                            name=col,
                            line=dict(width=3 if col in avg.keys() else 2),
                            opacity=1 if col in avg.keys() else 0.6,
                        ))

            fig.show()

        if config["plot_daily_returns"]:
            colors = [
                "hsl(" + str(h) + ",50%" + ",50%)"
                for h in np.linspace(0, 360, len(daily_returns.columns))
            ]

            fig2 = go.Figure(data=[
                go.Box(y=daily_returns[col], marker_color=colors[i], name=col)
                for i, col in enumerate(daily_returns.columns)
            ])

            fig2.update_layout(
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(zeroline=False, gridcolor="white"),
                paper_bgcolor="rgb(233,233,233)",
                plot_bgcolor="rgb(233,233,233)",
            )

            fig2.show()

def get_last_date(csv_filename):
    with open(csv_filename, 'r') as file:
        lines = file.readlines()
        # Iterate over lines in reverse order
        for line in reversed(lines):
            line = line.strip()
            if line:  # Check if the line is not empty
                last_date_str = line.split(',')[0]  # Assuming the date is the first field
                try:
                    return datetime.strptime(last_date_str, '%Y-%m-%d').date()
                except ValueError:
                    print(f"Invalid date format: {last_date_str}")
                    # Continue iterating if the date format is invalid
                    continue
    return None

def is_weekday(date):
    return date.weekday() < 5  # Monday to Friday are weekdays (0 to 4)

def is_after_4pm_est():
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    return now.hour >= 16  # Check if it's after 4 PM EST

def is_holiday(date):
    # Define a holiday calendar
    cal = USFederalHolidayCalendar()

    # Get the holidays for the year of the given date
    holidays = cal.holidays(start=date, end=date).to_pydatetime()    
    # Check if the input date is in the list of holidays    
    return date in holidays
    
def get_non_holiday_weekdays(start_date, end_date):            
    first_valid_date = start_date
    last_valid_date = end_date
    
    while not is_weekday(last_valid_date) or is_holiday(last_valid_date):
        last_valid_date -= timedelta(days=1)
    
    first_valid_date = start_date
    while first_valid_date >= end_date and \
          not is_weekday(first_valid_date) or is_holiday(first_valid_date):
        first_valid_date -= timedelta(days=1)

    return first_valid_date, last_valid_date

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
   
def calculate_start_end_dates(time):
    # Get today's date
    today = datetime.now().date()

    start_date_from_today = today - timedelta(days=time * 365)
    
    start_date, end_date = get_non_holiday_weekdays(start_date_from_today, today)

    # Return calculated dates
    return start_date, end_date

# MAIN
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
        
        last_date = get_last_date(sym_file)
        today = datetime.now().date()

        if not os.path.exists(sym_file):
            df_sym = get_stock_data(sym, start_date, end_date)
            df_sym.to_csv(sym_file)
        else:
            df_sym = pd.read_csv(sym_file, parse_dates=True, index_col="Date")
            first_date_index = df_sym.index[0]

            # Check if the first date index is greater than start_date
            if first_date_index > start_date:                
                df_sym = get_stock_data(sym, start_date, today)                
                df_sym.to_csv(sym_file)
            else:                          
                if last_date and last_date < today:                    
                    first_valid_date, last_valid_date = get_non_holiday_weekdays(last_date + timedelta(days=1), today)                    
                    if first_valid_date and last_valid_date:                        
                        if last_date < last_valid_date :                            
                            needs_update = True                                        
                            df_sym = update_store(sym, df_sym, first_valid_date, last_valid_date + timedelta(days=1))
                  

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
        # see whole df
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

            if holdings_match(result, symbols) and not needs_update:
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
                    sort_by_weights=config["sort_by_weights"],
                    optimization_model=optimization_method,
                    time_period=years,
                    minimum_weight=config["min_weight"],
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
            sort_by_weights=config["sort_by_weights"],
            optimization_model=optimization_method,
            time_period=years,
            minimum_weight=config["min_weight"],
        )

if len(stk) > 0:
    avg = stacked_output(stk)
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
        optimization_model=", ".join(sorted(list(set(sum(models.values(),
                                                         []))))),
        time_period=sorted_times[0],
        minimum_weight=min_weight,
        max_size=config["portfolio_max_size"],
    )

    # plotly graphs
    plot_graphs(daily_returns_to_plot, cumulative_returns_to_plot)

    # play sound when done
    if config["musicPath"]:
        try:
            playsound(config["musicPath"])
        except OSError:
            print('\ndone')

if __name__ == "__main__":
    cache_dir = "cache"
    max_age_hours = 72  # Adjust this value to your preference
    cleanup_cache(cache_dir, max_age_hours)