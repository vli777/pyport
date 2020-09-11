# pyport
portfolio optimization

optimization models available:
* inverse_variance
* min_volatility 
* max_sharpe
* efficient_risk
* efficient_return 
* max_return_min_volatility
* max_diversification
* max_decorrelation
* hierarchical risk parity
* hierarchical equal risk contribution
* online moving average reversion
* robust median reversion
* symmetric correlation driven nonparametric learning
* functional correlation driven nonparametric learning
* nested clustered optimization

Instructions:
1. Provide a single column csv containing the ticker symbols you want to includ
2. Specify the input file name (e.g. 'portfolio inputs')
3. Set the parameters in the `#config` section. Note that some optimization methods require additional parameters.
