# pyport
wip, portfolio optimization using the pypfopt module

optimization models available:
* sharpe
* min volatility
* max return for a target volatility
* min volatility for a target return
* black-litterman

to-do:
* hiearchical risk-parity
* omega ratio

Instructions:
1. Provide a list of symbols in portfolio optimization.py or in a column csv export. 
If using exported data from Yahoo Finance, place the csv in a folder.
2. Specify the folder and file names as needed.
3. Make sure all parameters required for the optimization type selected are updated before running e.g. BL needs prior views in a dictionary