import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pandas_datareader import data as pdr
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import re
import yfinance as yf
yf.pdr_override()
import csv

## config ##
input_file = 'max_inputs.csv'   
weight_bounds=(0, 1)        
l2_regularization = .1
show_discrete_share_allocation = False          
time_period_in_yrs = 1
starting_capital = 30000
symbols = []            
ignored_symbols = [

]
import_symbols_from_csv = True
import_data_from_csv = False
save_to_csv = True   
optimization_method = 'sharpe'         
optimization_config = {
    'sharpe': {},           # maximize return / volatility ratio
    'min vol': {},          # minimize portfolio variance
    'black': {              # black litterman incorprates your performance expectations
        'IWM': -1,
        'TLT': 0.5,
        'QQQ': 1
    },
    'target vol': 0.36,     # maximize return given a target volatility
    'target return': 1.00   # minimize volatility given a target return             
}   
## end config ##

# constants
FOLDER = '{}{}yr'.format(input_file, time_period_in_yrs)
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
            if name.upper() not in ignored_symbols:
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
    if import_data_from_csv:
        df_sym = pd.read_csv(PATH + '{}.csv'.format(sym), parse_dates=True, index_col="Date")
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
if optimization_method != 'black':
    ef.portfolio_performance(verbose=True)

# output
print('{} - {} ({} yrs)'.format(START_DATE, END_DATE, time_period_in_yrs))
print('portfolio allocation weights: ')
try: 
    clean_weights
except:
    clean_weights = ef.clean_weights()
for sym, weight in sorted(clean_weights.items()):
    if int(weight * 100) > 0:
        print(sym, '\t% 5.2f' %(weight))

# discrete share allocation
if show_discrete_share_allocation:
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=starting_capital)
    allocation, leftover = da.lp_portfolio()
    print('\ndiscrete share allocation given $', starting_capital)
    for sym in sorted(allocation):
        print(sym, '\t', allocation[sym])