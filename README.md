# pyport
portfolio optimization

optimization models available:
* min_volatility (CLA)
* max_sharpe (CLA2)
* efficient_risk (MVO)
* efficient_return  (MVO)
* hierarchical risk parity (HRP)
* hierarchical equal risk contribution (HERC)
* online moving average reversion (OLMAR)
* robust median reversion (RMR)
* symmetric correlation driven nonparametric learning (SCORN)
* functional correlation driven nonparametric learning (FCORNK)
* nested clustered optimization (NCO)

Instructions:
1. Create a config.yaml based on the testconfig.yaml template

2. In input file csv's, provide a single column containing the ticker symbols you want to include. # Commented lines will be skipped when downloading ticket data.

3. In the config file under input_files, you can list multiple files to include. Multiple entries can be used under models as well. The output will contain a simple average of all selections.