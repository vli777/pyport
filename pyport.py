from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation, ReturnsEstimators
import csv
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import re
import yfinance as yf
yf.pdr_override()

## config ##
input_file = 'watchlist.csv'
weight_bounds = (0, 1)
l2_regularization = 0
starting_capital = 100000
time_period_in_yrs = .72
symbols = []
ignored_symbols = [

]
update_freq = 7             # 0 always re-dl, or use import_data_from_csv false
import_symbols_from_csv = True
import_data_from_csv = True  # default uses local cache

optimization_method = 'herc'
# hrp
# herc
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
    'efficient_risk': 1.83,     # maximize return given a target volatility
    'efficient_return': 0.28,  # minimize volatility given a target return
    'risk_aversion': 10,
}
## end config ##

# constants
FOLDER = '{}yr'.format(time_period_in_yrs)
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
    import_data_from_csv = False
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
if len(symbols) == 0:
    import_symbols_from_csv = True
if import_symbols_from_csv and len(input_file) > 0:
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
        needs_refresh = time_elapsed > timedelta(days=update_freq)
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
if optimization_method in ['hrp', 'herc']:
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
else:
    temp = MeanVarianceOptimisation()
    expected_returns = ReturnsEstimators().calculate_mean_historical_returns(
        asset_prices=df, resample_by='D')
    covariance = ReturnsEstimators().calculate_returns(
        asset_prices=df, resample_by='D').cov()
    temp.allocate(asset_names=df.columns,
                  asset_prices=df,
                  expected_asset_returns=expected_returns,
                  covariance_matrix=covariance,
                  solution=optimization_method,
                  target_return=optimization_config['efficient_risk'],
                  target_risk=optimization_config['efficient_return'],
                  risk_aversion=optimization_config['risk_aversion'],
                  weight_bounds=weight_bounds)
    temp.get_portfolio_metrics()

temp_weights = temp.weights.sort_values(by=0, ascending=False, axis=1)
temp_weights = temp_weights.to_dict('records')
clean_weights = temp_weights[0]

# output
print('\n{} to {} ({} yrs)'.format(START_DATE, END_DATE, time_period_in_yrs))
print('optimization method:', optimization_method)
print('portfolio allocation weights: ')

for sym, weight in sorted(clean_weights.items(
), key=lambda kv: (kv[1], kv[0]), reverse=True):
    # if (int(weight * 100) >= 0.01):
    print(sym, '\t% 5.3f' % (weight))
