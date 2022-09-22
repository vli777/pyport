"""
Pyport - portfolio optimization
"""
import os
import csv
from datetime import datetime, timedelta
from collections import Counter
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def earlier_date(date_a, date_b, before=True):
    """
    return boolean if date_a < or > date_b
    """
    if isinstance(date_a, str):
        date_a = str_to_date(date_a)
    if isinstance(date_b, str):
        date_b = str_to_date(date_b)
    if before:
        return date_a < date_b
    else:
        return date_a > date_b


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


def get_stock_data(symbol, start_date, end_date, write=False):
    """
    download stock price data from yahoo finance with optional write to csv
    """
    print("Downloading {} {} - {} ...".format(symbol, start_date, end_date))
    symbol_df = yf.download(symbol, start=start_date, end=end_date)
    if write:
        symbol_df.to_csv(sym_file)
    return symbol_df


def update_store(symbol, df_symbol, target_start, target_end):
    """
    checks if the saved stock data is up to date, appends to symbol df and to saved csv
    returns appended df, bool if an update was made
    """
    update_status = False
    # check if we can append today's data after 4pm
    # if TODAY.weekday() < 5:
    #     days_until_refresh = 1
    # else:
    #     days_until_refresh = timedelta(
    #         days = 3 - TODAY.isoweekday() % 5).days

    # next_time_to_refresh = datetime.now().replace(
    #     hour=16, minute=5, second=0, microsecond=0) + timedelta(
    #     days=days_until_refresh)

    sym_filepath = PATH + "{}.csv".format(symbol)

    # mod_time = datetime.fromtimestamp(
    #         os.path.getmtime(sym_file)).replace(
    #         hour=16, minute=0, second=0, microsecond=0)

    # check if start and end dates are covered by the symbol data
    first_date = df_symbol.index[0]
    last_date = df_symbol.index[-1]
    if not isinstance(first_date, str):
        first_date_str = date_to_str(first_date)

    if earlier_date(target_start, first_date_str):
        # handle time selections starting on weekends
        if str_to_date(target_start).weekday() < 5 and earlier_date(
                target_start, first_date):
            appended_data = get_stock_data(symbol,
                                           target_start,
                                           first_date_str,
                                           write=False)
            df_symbol = appended_data.append(df_symbol)
            # save df to file
            df_symbol.to_csv(sym_filepath)
            # update change status
            update_status = True

    # if weekday, dl latest data & append to csv
    while target_end.weekday() >= 5:
        target_end -= timedelta(days=1)
    if target_end.hour < 16:
        target_end -= timedelta(days=1)

    adj_last_date = last_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    adj_target_end = target_end.replace(hour=0, minute=0, second=0, microsecond=0)

    if last_date.weekday() < 5 and earlier_date(adj_last_date, adj_target_end):
        appended_data = get_stock_data(symbol,
                                    last_date + timedelta(days=1),
                                    target_end,
                                    write=False)
        appended_data.reset_index(inplace=True)

        if appended_data.shape[0] > 0 and last_date != appended_data['Date'].iloc[0]:
            appended_data.to_csv(sym_filepath, mode="a", index=False, header=False)
            # update df
            appended_data = appended_data.set_index("Date")
            df_symbol = df_symbol.append(appended_data)
            # update change status
            update_status = True

    return df_symbol, update_status


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

    if (len(scaled) == 1 or any(np.isnan(val) for val in scaled.values()) or
            max(scaled.values()) < minimum_weight):
        scaled = {"SPY": 1}
    while len(scaled) > 0 and min(scaled.values()) < minimum_weight or len(scaled) > max_size:
        clipped = clip_by_weight(scaled, minimum_weight)
        scaled = scale_to_one(clipped)
        minimum_weight = get_min_by_size(scaled, max_size, minimum_weight)

    # store portfolio
    stk[optimization_model + str(time_period)] = [scaled, len(scaled)]
    if not os.path.exists(CWD + "cache"):
        os.makedirs(CWD + "cache")
    output_file = (
        CWD + "cache/" +
        "{}-{}-{}.csv".format(inputs, optimization_model, time_period))

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

        portfolio_returns = returns.apply(
            lambda row: apply_weights(row, list(scaled.values())),
            axis=1).sum(axis=1)

        cumulative_returns = portfolio_returns.add(1).cumprod()
        sharpe = sharpe_ratio(portfolio_returns)

        drawdown, time_under_water = drawdown_and_time_under_water(
            cumulative_returns)
        mdd, dd_time = drawdown.max(), time_under_water[drawdown.idxmax()] * 252

        print("\ntime period: {} to {} ({} yrs)".format(start_date, end_date,
                                                        time_period))
        print("inputs:", inputs)
        print("optimization methods:", optimization_model)
        print("sharpe ratio:", round(sharpe, 2))
        print("cumulative return: {}%".format(
            round(cumulative_returns[-1] * 100, 2)))
        print("drawdown: -{}, {} days to recover after {}".format(
            round(mdd, 2), round(dd_time, 2),
            drawdown.idxmax().date()))
        print("run on:", datetime.now())
        print(
            "portfolio allocation weights (min {:.2f}):".format(minimum_weight))
    else:
        print("max diversification recommended")

    if sort_by_weights:
        for symbol, weight in sorted(scaled.items(),
                                     key=lambda kv: (kv[1], kv[0]),
                                     reverse=True):
            print(symbol, "\t% 5.3f" % (weight))
    else:
        for symbol, weight in sorted(scaled.items()):
            print(symbol, "\t% 5.3f" % (weight))


def plot_graphs(data):
    """
    creates plotly graphs
    """
    if config["test_mode"]:
        print(data)
    if config["plot_cumulative_returns"] or config["plot_daily_returns"]:
        daily_returns = data.pct_change()[1:]
        if config["test_mode"]:
            print(daily_returns.head())

        cumulative_returns = daily_returns.add(1).cumprod().sub(1).mul(100)
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
                            mode="lines",
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


# MAIN
filtered_times = {k for k in config["models"].keys() if config["models"][k]}
sorted_times = sorted(filtered_times, reverse=True)

for times in sorted_times:
    if not config["models"][times]:
        continue

    START_DATE = date_to_str(TODAY +
                             relativedelta(months=-round(float(times) * 12)))
    END_DATE = date_to_str(TODAY + relativedelta(days=1))

    # get ticker symbols
    symbols = []
    for input_file in config["input_files"]:
        input_file += ".csv"
        with open(CWD + input_file) as file:
            for line in file:
                name = line.rstrip()
                # name = name.split('.')[0]
                if (not name.startswith("#") and name[:1].isalpha() and
                        name.upper()
                        not in [x.upper() for x in config["ignored_symbols"]]):
                    symbols.append(name.upper())
    symbols = sorted(set(symbols))
    if config["test_mode"]:
        print(symbols)

    # import data to df
    df = pd.DataFrame()
    DATA_UPDATED = False

    for sym in symbols:
        if not sym:
            continue
        sym_file = PATH + "{}.csv".format(sym)

        if not os.path.exists(sym_file) or (times == sorted_times[0] and
                                            not config["use_cached_data"]):
            df_sym = get_stock_data(sym, START_DATE, END_DATE, write=True)
        else:
            df_sym = pd.read_csv(sym_file, parse_dates=True, index_col="Date")
            df_sym, DATA_UPDATED = update_store(sym, df_sym, START_DATE, TODAY)

        df_sym.rename(columns={"Adj Close": sym}, inplace=True)
        df_sym.drop(["Open", "High", "Low", "Close", "Volume"],
                    axis=1,
                    inplace=True)

        # drop by time period
        # total_rows = df.shape[0]
        # pct_of_rows = float(times)
        df_sym = df_sym.loc[START_DATE::]

        # append symbol df to main df
        if df.empty:
            df = df_sym
        else:
            df = df.join(df_sym, how="outer")

    # nan fills in joined df
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)

    # store df for graphing
    if times == sorted_times[0]:
        dfs["data"] = df
        dfs["start"] = START_DATE
        dfs["end"] = END_DATE

    if config["test_mode"]:
        # see whole df
        df.to_csv("full_df.csv")
        df = df.head(int(len(df) * config["test_data_visible_pct"]))
        print(df.head(), df.tail(), "Null values present: ",
              df.isnull().values.any())

    # for each included model, run optimization
    for optimization in config["models"][times]:
        optimization_method = optimization.lower()

        print("\nCalculating {} {} allocation".format(
            times, optimization_method.upper()))

        INPUTS_LIST = ", ".join([str(i) for i in sorted(config["input_files"])])
        model_cache_file = CWD + "cache/{}-{}-{}.csv".format(
            INPUTS_LIST, optimization_method, times)

        if os.path.isfile(model_cache_file):
            with open(model_cache_file, newline="") as cached_data:
                reader = csv.reader(cached_data, delimiter=",")
                result = {row[0]: float(row[1]) for row in reader}

            if holdings_match(result, symbols) and not DATA_UPDATED:
                weights = pd.DataFrame(result,
                                       index=pd.RangeIndex(start=0,
                                                           stop=1,
                                                           step=1))
                output(
                    data=df,
                    allocation_weights=weights,
                    inputs=INPUTS_LIST,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    sort_by_weights=config["sort_by_weights"],
                    optimization_model=optimization_method,
                    time_period=times,
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
            start_date=START_DATE,
            end_date=END_DATE,
            sort_by_weights=config["sort_by_weights"],
            optimization_model=optimization_method,
            time_period=times,
            minimum_weight=config["min_weight"],
        )

if len(stk) > 0:
    avg = stacked_output(stk)
    sorted_avg = dict(sorted(avg.items(), key=lambda item: item[1]))
    min_weight = get_min_by_size(sorted_avg, config["portfolio_max_size"])
    models = {k: v for k, v in config["models"].items() if v is not None}

    output(
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
    plot_graphs(dfs["data"])
    # play sound when done
    if config["musicPath"]:
        try:
            playsound(config["musicPath"])
        except OSError:
            print('\ndone')
