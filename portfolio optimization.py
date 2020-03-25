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
weight_bounds=(0, 1)
write_to_csv = True

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
print(df.head())

# Calculate expected returns and sample covariance
mu = mean_historical_return(df)
cov_matrix = CovarianceShrinkage(df).ledoit_wolf()

# Optimise for maximal Sharpe ratio 
ef = EfficientFrontier(mu, cov_matrix, weight_bounds)
weights = ef.max_sharpe()

# Optimize Black Litterman
# relative views
# viewdict = { "QQQ": 0.10, "IWM": -0.10, "TLT": .10 }
# bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
# rets = bl.bl_returns()
# ef = EfficientFrontier(rets, cov_matrix)
# OR use market return-implied weights
# delta = black_litterman.market_implied_risk_aversion(df.SPY)
# bl.bl_weights(delta)
# cleaned_weights = bl.clean_weights()

# clean weights and write to csv
cleaned_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)
print(cleaned_weights)

if write_to_csv:
    ef.save_weights_to_file(PATH + DATE + ".csv")  # saves to file

# discrete share allocation
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=20000)
allocation, leftover = da.lp_portfolio()
for sym in sorted(allocation):
    print(sym, + allocation[sym])
print("Funds remaining: ${:.2f}".format(leftover))