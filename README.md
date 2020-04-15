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
1. Provide a single column list of ticker symbols column in csv format. 
OR You may download exported Yahoo Finance historical data manually and place them in a folder.
2. Specify the folder and input file name in the ##config section at the top
3. Make sure all parameters required for the optimization type selected are set before running e.g. BL needs prior views in a dictionary