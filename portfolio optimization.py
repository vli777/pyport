import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pandas_datareader import data as pdr
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import re
import yfinance as yf
yf.pdr_override()
import csv

## config ##
import_symbols_from_csv = True          # if not importing, use manually entered list
import_data_from_csv = False     # if using csv exports of yahoo finance data
save_to_csv = True              # saves a copy of imported yahoo finance data to csv  
folder = '6 mo'                         # optional: if your files are located in another folder
time_period_in_yrs = 0.5
input_file = 'portfolio inputs.csv'     # specify the input file name with file ext
symbols = []                            
ignored_symbols = ['SHY','IEF', 'TLT', 'IAU', 'GLD','INO', 'MRNA', 'GILD']                    
weight_bounds=(0, .33)          
starting_capital = 60000       
discrete_shares = True          # display whole number shares after allocation weights
optimization_method = 'sharpe'                                                  
# SHARPE - max return / volatility ratio
# BLACK - black litterman allows specifiying views on relative asset performance
viewdict = {    # if using BL, specify manual weights for each included asset to work properly
    'IWM': -1,
    'TLT': 0.5,
    'QQQ': 1
}
# TARGET VOL - max return given a target volatility
target_vol = .33                    
# TARGET RETURN min volatlity given a target return
target_return = .33             
## end config ##

# constants
DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")
CWD = os.getcwd() +'/'
PATH = CWD
if len(folder) > 0: PATH += folder +'/'
TODAY = datetime.today()
START_DATE = (TODAY + relativedelta(months=-round(time_period_in_yrs*12))).strftime('%Y-%m-%d')
END_DATE = (TODAY + relativedelta(days=1)).strftime('%Y-%m-%d')

# get ticker symbols 
if import_symbols_from_csv and len(input_file) > 0:
    with open(CWD + input_file) as file:
        for line in file:
            name = line.rstrip()
            if name not in ignored_symbols:
                symbols.append(name)
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
ef = EfficientFrontier(mu, cov_matrix, weight_bounds)
print ('\nStart Date:', START_DATE)
print ('End Date:', TODAY)
print ('Investment Horizon: {} YR'.format(time_period_in_yrs))

if optimization_method == 'sharpe':
    weights = ef.max_sharpe()
elif optimization_method == 'min vol':
    weights = ef.min_volatility()
elif optimization_method == 'target vol':
    weights = ef.efficient_risk(target_vol)
elif optimization_method == 'target return':
    weights = ef.efficient_return(target_return)
elif optimization_method == 'black':
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)    
    spx = pdr.get_data_yahoo('SPY', start=START_DATE, end=END_DATE)
    spx.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)
    delta = black_litterman.market_implied_risk_aversion(spx['Adj Close'])
    bl.bl_weights(delta)
    weights = bl.clean_weights()
    bl.portfolio_performance(verbose=True)
if optimization_method != 'black':
    ef.portfolio_performance(verbose=True)

# output
print('\nportfolio allocation weights: ')
clean_weights = ef.clean_weights()
for sym, weight in clean_weights.items():
    if int(weight * 100) > 0:
        print(sym, '\t% 5.2f' %(weight))

# discrete share allocation
if discrete_shares:
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=starting_capital)
    allocation, leftover = da.lp_portfolio()
    print('\ndiscrete share allocation given $', starting_capital)
    for sym in sorted(allocation):
        print(sym, '\t', allocation[sym])