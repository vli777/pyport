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

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

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
time_period_in_yrs = 1.83
min_weight_to_display = 0.02             
use_cached_data = True
sort_by_weights = True
reduce_reversion_weights = True
ignored_symbols = [             
    'ring', 'slvp'
    ]
models = [
    'herc', 'olmar', 'rmr'
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
        'plot_dendrogram': False,
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
                    "#") and name[:1].isalpha() and name.upper() not in [x.upper() for x in ignored_symbols]:
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
df = df.reindex(sorted(df.columns), axis=1)
# print(df.head(), df.isnull().values.any())

def output(weights, sort_by_weights=False, optimization_method=None):
    if isinstance(weights, dict):
        clean_weights = weights
    else: 
        clean_weights = weights.to_dict('records')[0]

    print('\n{} to {} ({} yrs)'.format(START_DATE, END_DATE, time_period_in_yrs))
    print('optimization method:', optimization_method)
    print('portfolio allocation weights: ')

    scaled = scale_to_one(clean_weights)
    clipped = { k: v for k, v in scaled.items() if v >= min_weight_to_display }
    scaled = scale_to_one(clipped)

    if sort_by_weights:
        for sym, weight in sorted(scaled.items(), 
            key=lambda kv: (kv[1], kv[0]), reverse=True # sort by weights
            ):
            print(sym, '\t% 5.3f' % (weight))
    else:
        for sym, weight in sorted(scaled.items()):
            print(sym, '\t% 5.3f' % (weight))
    
    if optimization_method != 'stack':
        stk[optimization_method] = scaled

def scale_to_one(weights):
        # scale to sum to 1
    total_alloc = sum(weights.values())
    scaled = { k: v / total_alloc for k, v in weights.items() if v / total_alloc >= min_weight_to_display }
    return scaled

stk = {}
temp = None

for optimization_method in models:
    print ('\nCalculating...', optimization_method)

    if (optimization_method == 'hrp'):
        temp = HierarchicalRiskParity()
        temp.allocate(
            asset_prices=df,
            linkage=optimization_config[optimization_method]['linkage'],
            )

    elif (optimization_method == 'herc'):
        temp = HierarchicalEqualRiskContribution()
        temp.allocate(
            asset_prices=df,
            risk_measure=optimization_config[optimization_method]['risk_measure'],
            linkage=optimization_config[optimization_method]['linkage'])
        
        if optimization_config[optimization_method]['plot_dendrogram']:
            z = temp.plot_clusters(assets=df.columns)
            plt.show(block=False)
        
    elif (optimization_method == 'cla'):
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
            alpha=optimization_config[optimization_method]['alpha']
        )
        temp.allocate(asset_prices=df, verbose=False)
        temp_dict = dict(zip(df.columns, temp.weights))
        temp.weights = temp_dict

    elif (optimization_method == 'rmr'):
        temp = RMR(
            epsilon=optimization_config[optimization_method]['epsilon'],
            n_iteration=optimization_config[optimization_method]['n_iteration'],
            window=optimization_config[optimization_method]['window']
        )
        temp.allocate(asset_prices=df, verbose=True)
        temp_dict = dict(zip(df.columns, temp.weights))
        temp.weights = temp_dict
        
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

    output(temp.weights, sort_by_weights, optimization_method)

if len(stk) > 1:
    t = [ v for k, v in stk.items() if k not in ['olmar', 'rmr'] ]
    total = sum(map(Counter, t), Counter())
    N = float(len(stk))
    avg = { k: v/N for k, v in total.items() }
    # to reduce the bias from high weights in the reversion series, they are scaled down 
    N2 = len(avg)
    for mr in ['olmar', 'rmr']:
        if mr in stk.keys():
            for k, v in stk[mr].items():
                if reduce_reversion_weights:
                    mod = v / N2
                else:
                    mod = v / N
                if k in avg.keys():
                    avg[k] += mod
                else:
                    avg[k] = mod
    print('input files:', input_files)
    output(avg, sort_by_weights = True, optimization_method = 'stack')
    
plt.show()