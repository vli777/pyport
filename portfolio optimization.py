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
import os
import re
import yfinance as yf
yf.pdr_override()
import csv

## config
folder = '1 yr'
input_file = 'portfolio inputs.csv'
symbols = []
import_csv_data = False     # if using exported yahoo finance data, set this to True
startDate = '2020-01-31'    
endDate = '2020-03-31'
weight_bounds=(0,1)         # (-1, 1) to include shorts
capital = 30000             # starting capital
opt = 'sharpe'              # black, min vol, target vol, target return

# opt method specific vars  
viewdict = {                # if using BL, need prior weights on each asset to work properly
    "TLT": 1,               # these weights are based on your belief of expected performance
    "IWM": -1 
}
if (opt == 'black'): 
    symbols = viewdict.keys()
max_vol = .33               # if maximizing return for target vol
target_return = .33         # if minimizing vol for a target return

## start main
DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")
PATH = os.getcwd() +'/' + folder +'/'

# get ticker symbols
if len(symbols) == 0:
    if import_csv_data:
        for file in os.listdir(PATH):
            print(file)
            if file.endswith(".csv") and not re.search(r'\d', file) and file != input_file :
                name = os.path.splitext(file)[0]
                symbols.append(name)  
    else:    
        with open(PATH + input_file) as file:
            for line in file:
                symbols.append(line.rstrip())

# Read in price data
df = pd.DataFrame()

for sym in symbols:    
    if not sym:
        continue
    if import_csv_data:
        df_sym = pd.read_csv(PATH + '{}.csv'.format(sym), parse_dates=True, index_col="Date")
    else:
        df_sym = pdr.get_data_yahoo(sym, start=startDate, end=endDate)

    df_sym.rename(columns={'Adj Close':sym}, inplace=True)
    df_sym.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

    if df.empty:
        df = df_sym
    else:
        df = df.join(df_sym, how='outer')

mu = mean_historical_return(df)
cov_matrix = CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(mu, cov_matrix, weight_bounds)

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
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=capital)
allocation, leftover = da.lp_portfolio()
print('\ndiscrete share allocation given $', capital)
for sym in sorted(allocation):
    print(sym, '\t', allocation[sym])