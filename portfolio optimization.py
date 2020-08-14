import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pandas_datareader import data as pdr
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os, time
import re
import yfinance as yf
yf.pdr_override()
import csv
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution

## config ##
input_file = 'temp.csv'   
weight_bounds=(0, 1)        
l2_regularization = 0
starting_capital = 100000       
time_period_in_yrs = .36
symbols = []  
ignored_symbols = [

]
update_freq = 7             # 0 always re-dl, or use import_data_from_csv false
import_symbols_from_csv = True
import_data_from_csv = True # default uses local cache
save_to_csv = True   
show_discrete_share_allocation = False
optimization_method = 'herc' 
optimization_config = {
    'sharpe': {},           # maximize return / volatility ratio
    'min vol': {},          # minimize portfolio variance
    'black': {              # black litterman incorprates your performance expectations
        'IWM': -1,
        'TLT': 0.5,
        'QQQ': 1
    },
    'hrp': {
        'linkage': 'average',
    },
    'herc': { 
        'risk_measure' : 'expected_shortfall',
        'linkage': 'average',
    },
    'target vol': 0.158,     # maximize return given a target volatility
    'target return': 1.67   # minimize volatility given a target return             
}   
## end config ##

# constants
FOLDER = '{}yr'.format(time_period_in_yrs)
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
    import_data_from_csv = False
CWD = os.getcwd() +'/'
PATH = CWD + FOLDER + '/'

DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")
TODAY = datetime.today()
START_DATE = (TODAY + relativedelta(months=-round(time_period_in_yrs*12))).strftime('%Y-%m-%d')
END_DATE = (TODAY + relativedelta(days=1)).strftime('%Y-%m-%d')

# get ticker symbols 
if len(symbols) == 0:
    import_symbols_from_csv = True
if import_symbols_from_csv and len(input_file) > 0:
    with open(CWD + input_file) as file:
        for line in file:
            name = line.rstrip()
            if not name.startswith("#") and name.upper() not in ignored_symbols:
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
    
    filepath = PATH + '{}.csv'.format(sym)
    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    time_elapsed = TODAY - mod_time.replace(hour=0, minute=0, second=0, microsecond=0)
    needs_refresh = time_elapsed > timedelta(days=update_freq)

    if import_data_from_csv and os.path.exists(FOLDER + '/' + sym + '.csv') and not needs_refresh:
        df_sym = pd.read_csv(filepath, parse_dates=True, index_col="Date")
    else:
        df_sym = pdr.get_data_yahoo(sym, start=START_DATE, end=END_DATE)
        if save_to_csv:
            df_sym.to_csv(PATH + '{}.csv'.format(sym))

    df_sym.rename(columns={'Adj Close':sym}, inplace=True)
    df_sym.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

    if df.empty:
        df = df_sym
    else:
        df = df.join(df_sym, how='outer')
df.fillna(method='bfill', inplace=True)
print (df.head())

# calculate optimal weights
mu = mean_historical_return(df)
cov_matrix = CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(mu, cov_matrix, weight_bounds, gamma=l2_regularization)

if optimization_method == 'sharpe':
    weights = ef.max_sharpe()
elif optimization_method == 'min vol':
    weights = ef.min_volatility()
elif optimization_method == 'target vol':
    weights = ef.efficient_risk(optimization_config[optimization_method])
elif optimization_method == 'target return':
    weights = ef.efficient_return(optimization_config[optimization_method])
elif optimization_method == 'black':
    bl = BlackLittermanModel(cov_matrix, absolute_views=optimization_config[optimization_method])    
    spx = pdr.get_data_yahoo('SPY', start=START_DATE, end=END_DATE)
    spx.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)
    delta = black_litterman.market_implied_risk_aversion(spx['Adj Close'])
    bl.bl_weights(delta)
    weights = clean_weights = bl.clean_weights()
    bl.portfolio_performance(verbose=True)
elif optimization_method in ['hrp', 'herc']:
    # Compute HRP weights
    if (optimization_method == 'hrp'):
        temp = HierarchicalRiskParity()
    elif (optimization_method == 'herc'):
        temp = HierarchicalEqualRiskContribution()
    print(optimization_config[optimization_method])
    temp.allocate(asset_prices=df, **optimization_config[optimization_method])
    temp_weights = temp.weights.sort_values(by=0, ascending=False, axis=1)
    temp_weights = temp_weights.to_dict('records')
    clean_weights = temp_weights[0]
if optimization_method not in ['black', 'hrp', 'herc']:
    ef.portfolio_performance(verbose=True)

# output
print('{} - {} ({} yrs)'.format(START_DATE, END_DATE, time_period_in_yrs))
print('portfolio allocation weights: ')
try: 
    clean_weights
except:
    clean_weights = ef.clean_weights()
for sym, weight in sorted(clean_weights.items(), key = lambda kv: (kv[1], kv[0]), reverse=True):
    if (int(weight * 100) >= 0.01):
        print(sym, '\t% 5.3f' %(weight))

# discrete share allocation
if show_discrete_share_allocation:
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=starting_capital)
    allocation, leftover = da.lp_portfolio()
    print('\ndiscrete share allocation given $', starting_capital)
    for sym in sorted(allocation):
        print(sym, '\t', allocation[sym])