import os
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import csv
from collections import Counter
import plotly.graph_objects as go
from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution
from portfoliolab.clustering.hrp import HierarchicalRiskParity
from portfoliolab.modern_portfolio_theory.mean_variance import MeanVarianceOptimisation, ReturnsEstimators
from portfoliolab.modern_portfolio_theory import CriticalLineAlgorithm
from portfoliolab.clustering.nco import NestedClusteredOptimisation
from portfoliolab.estimators.risk_estimators import RiskEstimators
from portfoliolab.online_portfolio_selection.rmr import RMR
from portfoliolab.online_portfolio_selection.olmar import OLMAR
from portfoliolab.online_portfolio_selection.fcornk import FCORNK
from portfoliolab.online_portfolio_selection.scorn import SCORN
from mlfinlab.backtest_statistics import sharpe_ratio, drawdown_and_time_under_water
import yfinance as yf
import yaml

# setup
config_filename = 'config.yaml'
with open(config_filename) as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
locals().update(config)

## global vars
stk = {}
avg = {}
dfs = None

TODAY = datetime.today()# - timedelta(days=30)

if not os.path.exists(folder):
    os.makedirs(folder)
CWD = os.getcwd() + '/'
PATH = CWD + folder + '/'

def earlier_date(a, b, before=True):    
    if before: 
        return a < b
    else:
        return a > b

def str_to_date(dateStr, fmt = '%Y-%m-%d'):
    return datetime.strptime(dateStr, fmt)    

def date_to_str(date, fmt = '%Y-%m-%d'):
    return date.strftime(fmt)

def get_stock_data(sym, start_date, end_date, write=False):    
    print ('Downloading {} {} - {} ...'.format(sym, start_date, end_date))
    df_sym = yf.download(
        sym,
        start=start_date,
        end=end_date)
    if write: df_sym.to_csv(sym_file)
    return df_sym


def update_data_store(df_sym, target_start):    
    # check if we can append today's data after 4pm
    # if TODAY.weekday() < 5:
    #     days_until_refresh = 1
    # else:
    #     days_until_refresh = timedelta(
    #         days = 3 - TODAY.isoweekday() % 5).days

    # next_time_to_refresh = datetime.now().replace(
    #     hour=16, minute=5, second=0, microsecond=0) + timedelta(
    #     days=days_until_refresh)

    sym_file = PATH + '{}.csv'.format(sym)     

    # mod_time = datetime.fromtimestamp(
    #         os.path.getmtime(sym_file)).replace(
    #         hour=16, minute=0, second=0, microsecond=0)

    # check if start and end dates are covered by the symbol data
    first_date = df_sym.index[0]
    last_date = df_sym.index[-1]
    if type(first_date) != str:
        first_date_str = date_to_str(first_date)
        last_date_str = date_to_str(last_date)
    
    if earlier_date(target_start, first_date_str):    
        # handle time selections starting on weekends        
        if str_to_date(target_start).weekday() < 5 and earlier_date(target_start, first_date):        
            print(target_start, first_date)
            appended_data = get_stock_data(sym, target_start, first_date_str, write=False)           
            df_sym = appended_data.append(df_sym)        
            # save df to file        
            df_sym.to_csv(sym_file)
    
    # if last date is earlier than Fri, dl latest data & append to csv
    if last_date.weekday() < 4 and earlier_date(last_date_str, date_to_str(TODAY)):                       
        appended_data = get_stock_data(sym, last_date + timedelta(days=1), TODAY, write=False)   
        appended_data.reset_index(inplace=True)         
        appended_data.to_csv(sym_file, mode='a', index=False, header=False)
        # update df
        appended_data = appended_data.set_index('Date')
        df_sym = df_sym.append(appended_data)

    return df_sym


def scale_to_one(weights):
    total_alloc = sum(weights.values())
    scaled = {
        k: v /
        total_alloc for k,
        v in weights.items()}
    return scaled


def custom_scaling(weights_dict, scaling):
    return {k: v * scaling for k, v in weights_dict.items()}


def stacked_output(stk):
    maxlen = max([v[1] for v in stk.values()])

    for model in stk:
        portfolio, n = stk[model]
        stk[model] = custom_scaling(weights_dict=portfolio, scaling=n / maxlen)

    t = [v for k, v in stk.items()]
    total = sum(map(Counter, t), Counter())
    avg = {k: v / len(stk) for k, v in total.items()}
    return avg


def apply_weights(row, weights):
    for i, _ in enumerate(row):
        row[i] *= weights[i]
    return row


def clip_by_weight(weights, min_weight):
    return {k: v for k, v in weights.items() if v > min_weight}


def get_min_by_size(weights, size, min_weight=0.01):
    if (len(weights) > size):
        sorted_weights = sorted(weights.values(), reverse=True)
        min_weight = sorted_weights[size]
    return min_weight


def holdings_match(cached_dict, symbols):
    for sym in cached_dict.keys():
        if sym not in symbols:
            print (sym, 'not found in', symbols)
            return False
    for sym in symbols:
        if sym not in cached_dict.keys():
            print (sym, 'not found in', cached_dict.keys())
            return False
    return True

def output(
    weights,
    inputs,
    start_date,
    end_date,
    max_size=21,
    sort_by_weights=False,
    optimization_method=None,    
    time_period=1.0,
    min_weight=0.01
    ):
    if isinstance(weights, dict):
        clean_weights = weights
    else:
        clean_weights = weights.to_dict('records')[0]
    if test_mode:
        print('raw weights', clean_weights)

    scaled = scale_to_one(clean_weights)

    if len(scaled) == 1 or any(
            np.isnan(val) for val in scaled.values()) or max(
            scaled.values()) < min_weight:
        scaled = {'SPY': 1}
    while (min(scaled.values()) < min_weight or len(scaled) > max_size):
        clipped = clip_by_weight(scaled, min_weight)
        scaled = scale_to_one(clipped)
        min_weight = get_min_by_size(scaled, max_size, min_weight=min_weight)

    # store portfolio
    stk[optimization_method + times] = [scaled, len(scaled)]
    if not os.path.exists(CWD + 'cache'):
        os.makedirs(CWD + 'cache')
    output_file = CWD + 'cache/' + \
        '{}-{}-{}.csv'.format(inputs, optimization_method, times)

    w = csv.writer(open(output_file, "w", newline=''))
    for key, val in scaled.items():
        w.writerow([key, val])
    filtered_symbols = [sym for sym in symbols if sym not in scaled.keys()]
    for sym in filtered_symbols:
        w.writerow([sym, 0])

    if len(scaled) > 0:
        portfolio = df[scaled.keys()]
        asset_returns = np.log(portfolio) - np.log(portfolio.shift(1))
        asset_returns = asset_returns.iloc[1:, :]

        portfolio_returns = asset_returns.apply(
            lambda row: apply_weights(
                row,
                list(scaled.values())
            ), axis=1).sum(axis=1)

        cumulative_returns = portfolio_returns.add(1).cumprod()
        sharpe = sharpe_ratio(portfolio_returns)

        dd, tuw = drawdown_and_time_under_water(cumulative_returns)
        mdd, dd_time = dd.max(), tuw[dd.idxmax()] * 252

        print(
            '\ntime period: {} to {} ({} yrs)'.format(
                start_date,
                end_date,
                time_period))
        print('inputs:', inputs)
        print('optimization methods:', optimization_method)
        print('sharpe ratio:', round(sharpe, 2))
        print('cumulative return: {}%'.format(
            round(cumulative_returns[-1] * 100, 2)))
        print('drawdown: -{}, {} days to recover after {}'.format(
            round(mdd, 2),
            round(dd_time, 2),
            dd.idxmax().date()
        ))
        print('run on:', datetime.now())
        print('portfolio allocation weights (min {:.2f}):'.format(min_weight))
    else:
        print('max diversification recommended')

    if sort_by_weights:
        for sym, weight in sorted(
            scaled.items(), key=lambda kv: (
                kv[1], kv[0]), reverse=True):
            print(sym, '\t% 5.3f' % (weight))
    else:
        for sym, weight in sorted(scaled.items()):
            print(sym, '\t% 5.3f' % (weight))

def plot_graphs(df):
    if test_mode:
        print(df)
    if plot_cumulative_returns or plot_daily_returns:
        daily_returns = df.pct_change()[1:]
        if test_mode:
            print(daily_returns.head())

        cumulative_returns = daily_returns.add(1).cumprod().sub(1).mul(100)
        if test_mode:
            print(cumulative_returns.head())

        if plot_cumulative_returns:
            if sort_by_weights:
                sorted_cols = cumulative_returns.sort_values(
                    cumulative_returns.index[-1],
                    ascending=False,
                    axis=1
                ).columns
                cumulative_returns = cumulative_returns[sorted_cols]

                fig = go.Figure()

                for col in cumulative_returns.columns:
                    fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns[col],
                        mode="lines",
                        name=col,
                        line=dict(width=3 if col in avg.keys() else 2),
                        opacity=1 if col in avg.keys() else 0.6,
                    ))

            fig.show()

        if plot_daily_returns:
            c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0,
                                                                        360, len(daily_returns.columns))]

            fig2 = go.Figure(data=[go.Box(
                y=daily_returns[col],
                marker_color=c[i],
                name=col
            ) for i, col in enumerate(daily_returns.columns)])

            fig2.update_layout(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False),
                yaxis=dict(
                    zeroline=False,
                    gridcolor='white'),
                paper_bgcolor='rgb(233,233,233)',
                plot_bgcolor='rgb(233,233,233)',
            )

            fig2.show()

### MAIN
sorted_times = sorted(models.keys(), reverse=True)
for times in sorted_times:
    if not models[times]:
        continue

    START_DATE = date_to_str(
        TODAY +
        relativedelta(
            months=-
            round(
                float(times) *
                12)))
    END_DATE = date_to_str(TODAY + relativedelta(days=1))

    # get ticker symbols
    symbols = []
    for input_file in input_files:
        input_file += '.csv'
        with open(CWD + input_file) as file:
            for line in file:
                name = line.rstrip()
                # name = name.split('.')[0]
                if not name.startswith("#") and name[:1].isalpha() and name.upper() not in [
                        x.upper() for x in ignored_symbols]:
                    symbols.append(name.upper())
    symbols = sorted(set(symbols))
    if test_mode: print(symbols)
    
    # import data to df
    df = pd.DataFrame()

    for sym in symbols:
        if not sym:
            continue
        sym_file = PATH + '{}.csv'.format(sym)
        
        if not os.path.exists(sym_file) or not use_cached_data and times == sorted_times[0]:    
            df_sym = get_stock_data(sym, START_DATE, END_DATE, write=True)
        else:    
            df_sym = pd.read_csv(sym_file, parse_dates=True, index_col="Date")       
            df_sym = update_data_store(df_sym, START_DATE)

        df_sym.rename(columns={'Adj Close': sym}, inplace=True)
        df_sym.drop(['Open', 'High', 'Low', 'Close','Volume'], axis=1, inplace=True)

        # drop by time period
        # total_rows = df.shape[0]
        # pct_of_rows = float(times)        
        df_sym = df_sym.loc[START_DATE::]

        # append symbol df to main df
        if df.empty:
            df = df_sym
        else:
            df = df.join(df_sym, how='outer')

    # nan fills in joined df
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)

    # store df for graphing
    if times == sorted_times[0]:
        dfs = df

    if test_mode:
        # see whole df
        df.to_csv('full_df.csv')
        df = df.head(int(len(df) * 0.72))
        print(
            df.head(),
            df.tail(),
            'Null values present: ',
            df.isnull().values.any())

    # for each included model, run optimization
    for optimization in models[times]:
        optimization_method = optimization.lower()
        temp = None

        print(
            '\nCalculating {} {} allocation'.format(
                times, optimization_method.upper()))

        inputs_list = ', '.join([str(i) for i in sorted(input_files)])
        model_cache_file = CWD + \
            'cache/{}-{}-{}.csv'.format(inputs_list,
                                        optimization_method, times)

        if os.path.isfile(model_cache_file):
            with open(model_cache_file, newline='') as data:
                reader = csv.reader(data, delimiter=',')
                result = {row[0]: float(row[1]) for row in reader}

            if holdings_match(
                result,
                symbols):
                continue_run = False
                weights = pd.DataFrame(
                    result, index=pd.RangeIndex(
                        start=0, stop=1, step=1))
                # send to output
                output(
                    weights=weights,
                    inputs=inputs_list,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    sort_by_weights=sort_by_weights,
                    optimization_method=optimization_method,
                    time_period=times,
                    min_weight=min_weight,
                )

                continue

        # model handling
        if (optimization_method == 'hrp'):
            temp = HierarchicalRiskParity()
            temp.allocate(
                asset_prices=df,
                linkage=optimization_config[optimization_method]['linkage'],
            )
        elif (optimization_method.find('herc') != -1):
            temp = HierarchicalEqualRiskContribution()
            temp.allocate(
                asset_prices=df,
                risk_measure=optimization_config[optimization_method]['risk_measure'],
                linkage=optimization_config[optimization_method]['linkage'])
        elif (optimization_method.find('nco') != -1):
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]
            temp = NestedClusteredOptimisation()
            if optimization_config[optimization_method]['sharpe']:
                mu_vec = np.array(asset_returns.mean())
            else:
                mu_vec = np.ones(len(df.columns))
            weights = temp.allocate_nco(
                asset_names=df.columns,
                cov=np.array(asset_returns.cov()),
                mu_vec=mu_vec.reshape(-1, 1)
            )
            temp.weights = weights
        elif (optimization_method.find('mc') != -1):
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :]
            temp = NestedClusteredOptimisation()
            mu_vec = np.array(asset_returns.mean())
            w_cvo, w_nco = temp.allocate_mcos(
                mu_vec=mu_vec.reshape(-1, 1),
                cov=np.array(asset_returns.cov()),
                num_obs=optimization_config[optimization_method]['num_obs'],
                num_sims=optimization_config[optimization_method]['num_sims'],
                kde_bwidth=optimization_config[optimization_method]['kde_bandwidth'],
                min_var_portf=not optimization_config[optimization_method]['sharpe'],
                lw_shrinkage=optimization_config[optimization_method]['lw_shrinkage']
            )
            w_nco = w_nco.mean(axis=0)
            temp_dict = dict(zip(df.columns, w_nco))
            temp.weights = temp_dict
        elif (optimization_method.find('cla') != -1):
            temp = CriticalLineAlgorithm()
            solution = optimization_config[optimization_method]['solution']
            temp.allocate(
                asset_prices=df,
                solution=solution)
        elif (optimization_method == 'olmar'):
            temp = OLMAR(
                reversion_method=optimization_config[optimization_method]['method'],
                epsilon=optimization_config[optimization_method]['epsilon'],
                window=optimization_config[optimization_method]['window'],
                alpha=optimization_config[optimization_method]['alpha'])
            temp.allocate(asset_prices=df, verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif (optimization_method == 'rmr'):
            temp = RMR(
                epsilon=optimization_config[optimization_method]['epsilon'],
                n_iteration=optimization_config[optimization_method]['n_iteration'],
                tau=optimization_config[optimization_method]['tau'],
                window=optimization_config[optimization_method]['window'])
            temp.allocate(
                asset_prices=df,
                verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif(optimization_method == 'scorn'):
            temp = SCORN(
                window=optimization_config[optimization_method]['window'],
                rho=optimization_config[optimization_method]['rho'])
            temp.allocate(
                asset_prices=df,
                resample_by=optimization_config[optimization_method]["resample"],
                verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif(optimization_method == 'fcornk'):
            temp = FCORNK(
                window=optimization_config[optimization_method]['window'],
                rho=optimization_config[optimization_method]['rho'],
                lambd=optimization_config[optimization_method]['lambd'],
                k=optimization_config[optimization_method]['k'],
            )
            temp.allocate(
                asset_prices=df,
                resample_by=optimization_config[optimization_method]["resample"],
                verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights[0]))
            temp.weights = temp_dict
        else:
            temp = MeanVarianceOptimisation()
            expected_returns = ReturnsEstimators(
            ).calculate_mean_historical_returns(asset_prices=df)
            covariance = ReturnsEstimators().calculate_returns(asset_prices=df).cov()
            temp.allocate(
                asset_names=df.columns,
                asset_prices=df,
                expected_asset_returns=expected_returns,
                covariance_matrix=covariance,
                solution=optimization_method,
                target_return=optimization_config['efficient_risk'],
                target_risk=optimization_config['efficient_return'],
                risk_aversion=optimization_config['risk_aversion'],
            )
            temp.get_portfolio_metrics()

        # send to output
        output(
            weights=temp.weights,
            inputs=inputs_list,
            start_date=START_DATE,
            end_date=END_DATE,
            sort_by_weights=sort_by_weights,
            optimization_method=optimization_method,
            time_period=times,
            min_weight=min_weight
        )

if len(stk) > 0:
    avg = stacked_output(stk)
    sorted_avg = dict(sorted(avg.items(), key=lambda item: item[1]))
    min_weight = get_min_by_size(sorted_avg, portfolio_max_size)
    models = {k: v for k, v in models.items() if v is not None}

    output(weights=sorted_avg,
           inputs=', '.join([str(i) for i in sorted(input_files)]),
           sort_by_weights=True,
           start_date=START_DATE,
           end_date=END_DATE,
           optimization_method=', '.join(sorted(list(set(sum(models.values(),
                                                             []))))),
           time_period=', '.join(models.keys()),
           min_weight=min_weight,
           max_size=portfolio_max_size
           )

    plot_graphs(dfs)