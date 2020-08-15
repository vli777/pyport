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

# Volume Bars
# volume = standard_data_structures.get_volume_bars('FILE_PATH', threshold=28000,
#                                               batch_size=1000000, verbose=False)

#vpin   
    # :param volume: (pd.Series) Bar volume
    # :param buy_volume: (pd.Series) Bar volume classified as buy (either tick rule, BVC or aggressor side methods applied)
    # :param window: (int) Estimation window
    # :return: (pd.Series) VPIN series

## config ##
input_file = 'fngs.csv'
time_period_in_yrs = 2.87       
ignored_symbols = [             # use this to filter out symbols in a csv input file

]
min_alloc = 0                # don't output weights below this value
update_freq_days = 7            # 0 to always dl latest data
optimization_method = 'herc'
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
    'olmar':{                  # SMA
        'epsilon': 10,          # reversion threshold
        'window': 11,            # SMA window
    },
    'rmr':{
        'epsilon': 13.37,
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
with open(CWD + input_file) as file:
    for line in file:
        name = line.rstrip()
        if not name.startswith(
                "#") and name.upper() not in ignored_symbols:
            symbols.append(name.upper())
            if optimization_method == 'black':
                if name not in optimization_config[optimization_method]:
                    optimization_config[optimization_method][name] = 1
symbols = list(set(symbols))

# Read in price data
df = pd.DataFrame()
for sym in symbols:
    if not sym:
        continue

    sym_file = PATH + '{}.csv'.format(sym)

    try:
        mod_time = datetime.fromtimestamp(os.path.getmtime(sym_file))
        time_elapsed = TODAY - \
            mod_time.replace(hour=0, minute=0, second=0, microsecond=0)
        needs_refresh = time_elapsed > timedelta(days=update_freq_days)
    except BaseException:
        needs_refresh = True

    if needs_refresh:
        print(
            '{} local data cache out of date. downloading latest price data...'.format(sym))
        df_sym = pdr.get_data_yahoo(sym, start=START_DATE, end=END_DATE)
        df_sym.to_csv(sym_file)
    else:
        df_sym = pd.read_csv(sym_file, parse_dates=True, index_col="Date")

    df_sym.rename(columns={'Adj Close': sym}, inplace=True)
    df_sym.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

    if df.empty:
        df = df_sym
    else:
        df = df.join(df_sym, how='outer')

df.fillna(method='bfill', inplace=True)
# print (df.head())

# calculate optimal weights
if optimization_method in ['hrp', 'herc', 'cla', 'olmar', 'rmr']:
    # Compute HRP weights
    if (optimization_method == 'hrp'):
        temp = HierarchicalRiskParity()
        temp.allocate(
            asset_prices=df,
            linkage=optimization_config[optimization_method]['linkage'])
    elif (optimization_method == 'herc'):
        temp = HierarchicalEqualRiskContribution()
        temp.allocate(
            asset_prices=df,
            risk_measure=optimization_config[optimization_method]['risk_measure'],
            linkage=optimization_config[optimization_method]['linkage'])
    elif (optimization_method == 'cla'):
        temp = CriticalLineAlgorithm()
        solution = optimization_config[optimization_method]['solution']
        temp.allocate(
            asset_prices=df,
            solution=solution)

        if solution == 'max_sharpe':
            print('max sharpe', temp.max_sharpe)
        elif solution == 'min_volatility':
            print('min variance', temp.min_var)
        elif solution == 'efficient_frontier':
            print('means', temp.efficient_frontier_means, 'sigma', temp.efficient_frontier_sigma)
    elif (optimization_method == 'olmar'):
        temp = OLMAR(
            reversion_method=1, 
            epsilon=optimization_config[optimization_method]['epsilon'],
            window=optimization_config[optimization_method]['window'],
        )
        temp.allocate(asset_prices=df, verbose=True)
        temp.weights = temp.all_weights
    elif (optimization_method == 'rmr'):
        temp = RMR(
            epsilon=optimization_config[optimization_method]['epsilon'],
            n_iteration=optimization_config[optimization_method]['n_iteration'],
            window=optimization_config[optimization_method]['window']
        )
        temp.allocate(asset_prices=df, verbose=True)
        temp.weights = temp.all_weights
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
    
temp_weights = temp.weights #.sort_values(by=0, ascending=False, axis=1)
clean_weights = temp_weights.to_dict('records')[0]

# output
print('\n{} to {} ({} yrs)'.format(START_DATE, END_DATE, time_period_in_yrs))
print('optimization method:', optimization_method)
print('portfolio allocation weights: ')

for sym, weight in sorted(clean_weights.items(
), key=lambda kv: (kv[1], kv[0]), reverse=True):
    if (weight >= min_alloc):
        print(sym, '\t% 5.3f' % (weight))
