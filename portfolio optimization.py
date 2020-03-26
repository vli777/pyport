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

## Config
folder = '1 yr'
weight_bounds=(-1,1)
capital = 20000
max_vol = .33
target_return = .33
viewdict = { "QQQ": 0.10, "TLT": .10 }
use_bl_views = False # uses market return derived weights
opt = '' # black, min vol, target vol, target return

## start main
DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")
PATH = os.getcwd() +'/' + folder +'/'

# get ticker symbols
symbols = []
for file in os.listdir(PATH):
    if file.endswith(".csv") and not re.search(r'\d', file):
        name = os.path.splitext(file)[0]
        symbols.append(name)  

# Read in price data
df = pd.DataFrame()

for sym in symbols:
    df_sym = pd.read_csv(PATH + '{}.csv'.format(sym), parse_dates=True, index_col="Date")
    df_sym.rename(columns={'Adj Close':sym}, inplace=True)
    df_sym.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

    if df.empty:
        df = df_sym
    else:
        df = df.join(df_sym, how='outer')

mu = mean_historical_return(df)
cov_matrix = CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(mu, cov_matrix, weight_bounds)

def getBLweights(viewdict, use_bl_views):
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, cov_matrix)
    delta = black_litterman.market_implied_risk_aversion(df.SPY)
    bl.bl_weights(delta)
    return bl.clean_weights()

if opt == 'sharpe':
    weights = ef.max_sharpe()
elif opt == 'min vol':
    weights = ef.min_volatility()
elif opt == 'target vol':
    weights = ef.efficient_risk(max_vol)
elif opt == 'target return':
    weights = ef.efficient_return(target_return)
else:
    weights = getBLweights(viewdict, use_bl_views)

# discrete share allocation
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=capital)
allocation, leftover = da.lp_portfolio()
for sym in sorted(allocation):
    print(sym, + allocation[sym])
print("Funds remaining: ${:.2f}".format(leftover))