from mlfinlab.backtest_statistics import sharpe_ratio
import plotly.graph_objects as go
import numpy as np
import sys
from portfolio import *


def output_results(df, weights, model_name, config, start_date, end_date, years):
    output(
        data=df,
        allocation_weights=weights,
        inputs=", ".join([str(i) for i in sorted(config.input_files)]),
        start_date=start_date,
        end_date=end_date,        
        optimization_model=model_name,
        time_period=years,
        minimum_weight=config.config["min_weight"],
        max_size=config.config.get("portfolio_max_size", 10),
        config=config
    )
    

def output(data, allocation_weights, inputs, start_date, end_date, 
           max_size=10, optimization_model=None, time_period=1.0, 
           minimum_weight=0.01, config=None):
    """Produces console output with descriptive statistics of the provided portfolio."""
    
    # Ensure allocation_weights is a dictionary
    clean_weights = allocation_weights if isinstance(allocation_weights, dict) else dict(allocation_weights)
    
    if config and config.test_mode:
        print("Raw weights:", clean_weights)

    # If the number of items exceeds max_size, remove the smallest ones
    if len(clean_weights) > max_size:
        # Sort by weight (ascending) and keep the largest `max_size` items
        sorted_weights = sorted(clean_weights.items(), key=lambda kv: kv[1], reverse=False)
        
        # Select the largest max_size weights
        limited_weights = dict(sorted_weights[-max_size:])
        
        # Reassign to clean_weights, no need to renormalize since it's already done
        clean_weights = limited_weights

    # Proceed with portfolio and calculations
    if len(clean_weights) == 0:
        print("Max diversification recommended")
        sys.exit()

    portfolio = data[clean_weights.keys()]
    returns = np.log(portfolio) - np.log(portfolio.shift(1))
    returns = returns.iloc[1:, :]  # Remove NaN row after log
    weights_vector = list(clean_weights.values())

    weighted_returns = returns.mul(weights_vector)
    portfolio_returns = weighted_returns.sum(axis=1)
    portfolio_cumulative_returns = (portfolio_returns + 1).cumprod()

    try:
        sharpe = sharpe_ratio(portfolio_returns)
    except ZeroDivisionError:
        sharpe = 0

    print(f"\nTime period: {start_date} to {end_date} ({time_period} yrs)")
    print("Optimization method:", optimization_model)
    print("Sharpe ratio:", round(sharpe, 2))
    print(f"Cumulative return: {round((portfolio_cumulative_returns[-1] - 1) * 100, 2)}%")
    print(f"Portfolio allocation weights (min {minimum_weight:.2f}):")

    sorted_weights = sorted(clean_weights.items(), key=lambda kv: kv[1], reverse=config.sort_by_weights)

    for symbol, weight in sorted_weights:
        print(f"{symbol} \t{weight:.3f}")

    # Prepare the dataframes for returns and cumulative returns
    portfolio_returns = portfolio_returns.to_frame(name='SIM_PORT')
    portfolio_cumulative_returns = portfolio_cumulative_returns.to_frame(name='SIM_PORT')

    all_daily_returns = returns.join(portfolio_returns)
    all_cumulative_returns = ((portfolio_cumulative_returns) - 1).join(returns.add(1).cumprod() - 1)

    return all_daily_returns, all_cumulative_returns

