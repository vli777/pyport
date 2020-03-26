# pyport

wip, portfolio optimization using the pypfopt module

optimization models available:
* sharpe
* black-litterman

to-do:
* hiearchical risk-parity
* omega ratio
* convert optimization code blocks to callable functions

Instructions:
1. Download [Yahoo] stock price data and place in a folder within the root directory.
2. Specify folder name and other config params
3. Run

If using Black Litterman model, you need to enter relative views for each asset.
Standard columns OHLC and volume are dropped. Date as index, Adj Close as price input

