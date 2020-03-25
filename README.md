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
2. Specify folder name and configure allocation limits in weight_bounds.
3. Run

Standard columns OHLC and volume are dropped. Date as index, Adj Close as price input

