from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation, ReturnsEstimators
from mlfinlab.portfolio_optimization import CriticalLineAlgorithm
from mlfinlab.online_portfolio_selection.rmr import RMR 
from mlfinlab.online_portfolio_selection.olmar import OLMAR
from mlfinlab.online_portfolio_selection.fcornk import FCORNK
from mlfinlab.online_portfolio_selection.scorn import SCORN
from mlfinlab.microstructural_features.third_generation import get_vpin
from mlfinlab.data_structures import standard_data_structures

from collections import Counter
import csv
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import re
import yfinance as yf
yf.pdr_override()

# scorn window 16, rho .21
# fcornk window 1.83, rho .87

## config ##
input_files = [
    'sectors'
    ]
time_period_in_yrs = 0.72
min_weight_to_display = 0.03                
use_cached_data = True

ignored_symbols = [             
    
    ]
models = [
    'cla', 'herc', 'olmar', 'rmr'
    ]
# rmr
# olmar
# herc
# hrp
# cla
# inverse_variance
# min_volatility
# max_sharpe
# efficient_risk
# efficient_return
# max_return_min_volatility
# max_diversification
# max_decorrelation

optimization_config = {
    'hrp': {
        'linkage': 'single',
    },
    'herc': {
        'risk_measure': 'conditional_drawdown_risk',
        'linkage': 'ward',
    },
    'olmar':{                  
        'method': 1,            # 1 for SMA, 2 for EWA    
        'epsilon': 11, #23        # reversion threshold
        'window': 11,           
        'alpha': 0.11,
    },
    'rmr':{
        'epsilon': 14,
        'n_iteration': 237,
        'window': 21,
    },
    'cla': { 'solution': 'max_sharpe' },
    'efficient_risk': 1.83,     # maximize return given a target volatility
    'efficient_return': 0.28,   # minimize volatility given a target return
    'risk_aversion': 10,
}
## end config ##

# constants
FOLDER = '{}yr'.format(time_period_in_yrs)
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
CWD = os.getcwd() + '/'
PATH = CWD + FOLDER + '/'

DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")
TODAY = datetime.today()
START_DATE = (
    TODAY +
    relativedelta(
        months=-
        round(
            time_period_in_yrs *
            12))).strftime('%Y-%m-%d')
END_DATE = (TODAY + relativedelta(days=1)).strftime('%Y-%m-%d')

# get ticker symbols
symbols = []
for input_file in input_files:
    input_file += '.csv'
    with open(CWD + input_file) as file:
        for line in file:
            name = line.rstrip()
            name=name.split('.')[0]
            if not name.startswith(
                    "#") and name[:1].isalpha() and name.upper() not in ignored_symbols:
                symbols.append(name.upper())
symbols = list(set(symbols))
# print(symbols)

def get_stock_data(sym):
    df_sym = pdr.get_data_yahoo(sym, start=START_DATE, end=END_DATE)
    df_sym.to_csv(sym_file)
    return df_sym

# Read in price data
df = pd.DataFrame()
for sym in symbols:
    if not sym:
        continue
    sym_file = PATH + '{}.csv'.format(sym)

    if use_cached_data:
        try:
            mod_time = datetime.fromtimestamp(os.path.getmtime(sym_file))
            time_elapsed = TODAY - \
                mod_time.replace(hour=0, minute=0, second=0, microsecond=0)
            needs_refresh = time_elapsed > timedelta(days=max(7,int(0.07 * time_period_in_yrs)))
        except BaseException:
            needs_refresh = True

    if needs_refresh:
        print('{} local data cache out of date. downloading latest price data...'.format(sym))
        df_sym = get_stock_data(sym)
    else:
        df_sym = pd.read_csv(sym_file, parse_dates=True, index_col="Date")
        if df_sym.empty:
            df_sym = get_stock_data(sym)

    df_sym.rename(columns={'Adj Close': sym}, inplace=True)
    df_sym.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

    if df.empty:
        df = df_sym
    else:
        df = df.join(df_sym, how='outer')

df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)
# print(df.head(), df.tail(), df.isnull().values.any())

def output(weights):
    if isinstance(weights, dict):
        clean_weights = weights
    else: 
        weights = weights.sort_values(by=0, ascending=False, axis=1)    
        clean_weights = weights.to_dict('records')[0]

    stk.append(clean_weights)

    print('\n{} to {} ({} yrs)'.format(START_DATE, END_DATE, time_period_in_yrs))
    print('optimization method:', optimization_method)
    print('portfolio allocation weights: ')

    for sym, weight in sorted(clean_weights.items(
    ), key=lambda kv: (kv[1], kv[0]), reverse=True):
        if (weight >= min_weight_to_display):
            print(sym, '\t% 5.3f' % (weight))

stk = []

for optimization_method in models:
    print ('\nCalculating...', optimization_method)
    # calculate optimal weights
    if optimization_method in ['hrp', 'herc', 'cla', 'olmar', 'rmr']:
        # Compute HRP weights
        if (optimization_method == 'hrp'):
            temp = HierarchicalRiskParity()
            temp.allocate(
                asset_prices=df,
                linkage=optimization_config[optimization_method]['linkage'])
            output(temp.weights)
        elif (optimization_method == 'herc'):
            temp = HierarchicalEqualRiskContribution()
            temp.allocate(
                asset_prices=df,
                risk_measure=optimization_config[optimization_method]['risk_measure'],
                linkage=optimization_config[optimization_method]['linkage'])
            output(temp.weights)
        elif (optimization_method == 'cla'):
            temp = CriticalLineAlgorithm()
            solution = optimization_config[optimization_method]['solution']
            temp.allocate(
                asset_prices=df,
                solution=solution)
            output(temp.weights)
        elif (optimization_method == 'olmar'):
            temp = OLMAR(
                reversion_method=optimization_config[optimization_method]['method'], 
                epsilon=optimization_config[optimization_method]['epsilon'],
                window=optimization_config[optimization_method]['window'],
                alpha=optimization_config[optimization_method]['alpha']
            )
            temp.allocate(asset_prices=df, verbose=True)
            temp_dict = dict(zip(df.columns, temp.weights))
            output(temp_dict)
        elif (optimization_method == 'rmr'):
            temp = RMR(
                epsilon=optimization_config[optimization_method]['epsilon'],
                n_iteration=optimization_config[optimization_method]['n_iteration'],
                window=optimization_config[optimization_method]['window']
            )
            temp.allocate(asset_prices=df, verbose=True)
            temp_dict = dict(zip(df.columns, temp.weights))
            output(temp_dict)
        else:
            temp = MeanVarianceOptimisation()
            expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=df)
            covariance = ReturnsEstimators().calculate_returns(asset_prices=df).cov()
            temp.allocate(asset_names=df.columns,
                        asset_prices=df,
                        expected_asset_returns=expected_returns,
                        covariance_matrix=covariance,
                        solution=optimization_method,
                        target_return=optimization_config['efficient_risk'],
                        target_risk=optimization_config['efficient_return'],
                        risk_aversion=optimization_config['risk_aversion'],
                        )
            temp.get_portfolio_metrics()
            output(temp.weights)

if len(models) > 1:
    total = sum(map(Counter, stk), Counter())
    N = float(len(stk))
    stk = { k: v/N for k, v in total.items() if v/N >= min_weight_to_display }
    total_alloc = sum(stk.values())
    scaled = { k: v / total_alloc for k, v in stk.items() }

    print('\n{} to {} ({} yrs)'.format(START_DATE, END_DATE, time_period_in_yrs))
    print('input files:', input_files)
    print('optimization method: STACK', models)
    print('portfolio allocation weights: ')

    for sym, weight in sorted(scaled.items(
    ), key=lambda kv: (kv[1], kv[0]), reverse=True):
        print(sym, '\t% 5.3f' % (weight))

