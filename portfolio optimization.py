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
folder = '1 yr'                         # optional: if your files are located in another folder
input_file = 'portfolio inputs.csv'     # specify the input file name with file ext
symbols = []                            # manual symbol list
ignored_symbols = ['GLD','ZM']          # symbols in this list will not be included in optimization  

# optimization parameters
yrs = 0.5                 # investment horizon / lookback period
weight_bounds=(0, 1)      # e.g. (-1, 1) includes shorts
capital = 30000           # starting capital
opt = 'sharpe'            # sharpe, black, min portfolio vol, target vol (max return), target return (minimizes vol)

# other options
import_data_from_csv = True     # if using csv exports of yahoo finance data
save_to_csv = True               # saves a copy of imported yahoo finance data to csv  
use_bonds = False         # includes ETF proxies for short, medium, long term bonds  
discrete_shares = True    # display whole number shares after allocation weights

# BLACK LITTERMAN RELATIVE VIEWS
viewdict = {              # if using BL, need prior weights on each asset to work properly
    "TLT": 1,             # these weights are based on your belief of expected performance
    "IWM": -1 
}
if (opt == 'black'): 
    symbols = viewdict.keys()

# MAXIMUM RETURN GIVEN TARGET VOLATILITY
max_vol = .33             # if maximizing return for target vol

# MINIMIZE VOLATILITY GIVEN TARGET RETURN
target_return = .33       # if minimizing vol for a target return

## end config ##

DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")
CWD = os.getcwd() +'/'
PATH = CWD
if len(folder) > 0:
    PATH += folder +'/'
TODAY = datetime.today()
startDate = (TODAY + relativedelta(months=-round(yrs*12))).strftime('%Y-%m-%d')
endDate = TODAY.strftime('%Y-%m-%d')
if not use_bonds:
    ignored_symbols += ['SHY', 'IEF', 'TLT'] 

# get ticker symbols 
if import_symbols_from_csv and len(input_file) > 0:
    symbols = []
    with open(CWD + input_file) as file:
        for line in file:
            name = line.rstrip()
            if name not in ignored_symbols:
                symbols.append(name)

# Read in price data
df = pd.DataFrame()
for sym in symbols:    
    if not sym:
        continue
    if import_data_from_csv:
        df_sym = pd.read_csv(PATH + '{}.csv'.format(sym), parse_dates=True, index_col="Date")
    else:
        df_sym = pdr.get_data_yahoo(sym, start=startDate, end=endDate)
        if save_to_csv:
            df_sym.to_csv(PATH + '{}.csv'.format(sym))

    df_sym.rename(columns={'Adj Close':sym}, inplace=True)
    df_sym.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

    if df.empty:
        df = df_sym
    else:
        df = df.join(df_sym, how='outer')

mu = mean_historical_return(df)
cov_matrix = CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(mu, cov_matrix, weight_bounds)
print ('\nStart Date:', startDate)
print ('End Date:', endDate)
print ('Investment Horizon: {} YR'.format(yrs))

if opt == 'sharpe':
    weights = ef.max_sharpe()
elif opt == 'min vol':
    weights = ef.min_volatility()
elif opt == 'target vol':
    weights = ef.efficient_risk(max_vol)
elif opt == 'target return':
    weights = ef.efficient_return(target_return)
elif opt == 'black':
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)    
    spx = pdr.get_data_yahoo('SPY', start=startDate, end=endDate)
    spx.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)
    delta = black_litterman.market_implied_risk_aversion(spx['Adj Close'])
    bl.bl_weights(delta)
    weights = bl.clean_weights()
    bl.portfolio_performance(verbose=True)

if opt != 'black':
    ef.portfolio_performance(verbose=True)

print('\nportfolio allocation weights: ')
clean_weights = ef.clean_weights()
for sym, weight in clean_weights.items():
    if int(weight * 100) > 0:
        print(sym, '\t% 5.2f' %(weight))

# discrete share allocation
if discrete_shares:
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=capital)
    allocation, leftover = da.lp_portfolio()
    print('\ndiscrete share allocation given $', capital)
    for sym in sorted(allocation):
        print(sym, '\t', allocation[sym])