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
1. Provide a single column csv containing the ticker symbols you want to include in portfolio optimization
OR download exported historical price data from Yahoo Finance and set `import_csv_data` to True.
2. Specify the input file name and folder name if your files are not in the root directory.
3. Set the parameters in the `#config` section. Some optimization methods require additional parameters.
