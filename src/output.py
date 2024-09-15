from mlfinlab.backtest_statistics import sharpe_ratio
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys
import csv
from portfolio import *

def save_model_results(model_name, time_period, input_filename, symbols, scaled):
    """Saves the model results to cache."""
    cache_dir = Path.cwd() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_file = cache_dir / f"{input_filename}-{model_name}-{time_period}.csv"
    
    print('Saving results to model cache')
    with output_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, val in scaled.items():
            writer.writerow([key, val])

        filtered_symbols = [sym for sym in symbols if sym not in scaled.keys()]
        for symbol in filtered_symbols:
            writer.writerow([symbol, 0])


def output_results(df, weights, config, start_date, end_date, symbols, years):
    output(
        data=df,
        allocation_weights=weights,
        inputs=", ".join([str(i) for i in sorted(config.input_files)]),
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        optimization_model=", ".join(sorted(list(set(sum(config.models.values(), []))))),
        time_period=years,
        minimum_weight=config.config["min_weight"],
        max_size=config.config.get("portfolio_max_size", 10),
        config=config
    )
    

def output(data, allocation_weights, inputs, start_date, end_date, symbols, 
           max_size=21, optimization_model=None, time_period=1.0, 
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

    # Save the results to the model cache
    save_model_results(model_name=optimization_model, time_period=time_period,
                       input_filename=inputs, symbols=symbols, scaled=clean_weights)

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
    print("Inputs:", inputs)
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


def plot_graphs(daily_returns, cumulative_returns, avg, config):
    """Creates Plotly graphs for daily and cumulative returns."""
    if config.test_mode:
        print(daily_returns, cumulative_returns)

    if config.plot_daily_returns:
        colors = ["hsl(" + str(h) + ",50%,50%)" for h in np.linspace(0, 360, len(daily_returns.columns))]
        fig2 = go.Figure([go.Box(y=daily_returns[col], marker_color=colors[i], name=col)
                          for i, col in enumerate(daily_returns.columns)])
        fig2.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(zeroline=False, gridcolor="white"),
            paper_bgcolor="rgb(233,233,233)",
            plot_bgcolor="rgb(233,233,233)",
            hoverlabel=dict(font=dict(size=22)),
        )
        fig2.show()

    if config.plot_cumulative_returns:
        fig = go.Figure()
        hovertemplate = '%{x} <b>%{y:.0%}</b>'
        
        if config.sort_by_weights:
            sorted_cols = cumulative_returns.sort_values(
                cumulative_returns.index[-1], ascending=False, axis=1).columns
            cumulative_returns = cumulative_returns[sorted_cols]

        for col in cumulative_returns.columns:
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index, y=cumulative_returns[col],
                mode="lines" if col != "SIM_PORT" else "lines+markers",
                name=col, line=dict(width=3 if col in avg.keys() else 2),
                opacity=1 if col in avg.keys() else 0.6,
            ))

        fig.update_layout(hoverlabel=dict(font=dict(size=22)))
        fig.update_traces(hovertemplate=hovertemplate)
        fig.show()
