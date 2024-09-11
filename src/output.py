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


def scale_and_clip_weights(weights, min_weight, max_size):
    """Scales and clips the weights to fit the minimum weight and max size constraints."""
    scaled = scale_to_one(weights)
    while len(scaled) > 0 and (min(scaled.values()) < min_weight or len(scaled) > max_size):
        clipped = clip_by_weight(scaled, min_weight)
        scaled = scale_to_one(clipped)
        min_weight = get_min_by_size(scaled, max_size, min_weight)
    return scaled


def output(data, allocation_weights, inputs, start_date, end_date, symbols, stack, 
           max_size=21, optimization_model=None, time_period=1.0, 
           minimum_weight=0.01, config=None):
    """Produces console output with descriptive statistics of the provided portfolio."""
    
    clean_weights = allocation_weights if isinstance(allocation_weights, dict) else dict(allocation_weights)
    
    if config and config.test_mode:
        print("Raw weights:", clean_weights)

    scaled = scale_and_clip_weights(clean_weights, minimum_weight, max_size)
    stack[optimization_model + str(time_period)] = [scaled, len(scaled)]
    
    save_model_results(model_name=optimization_model, time_period=time_period,
                       input_filename=inputs, symbols=symbols, scaled=scaled)

    if len(scaled) == 0:
        print("Max diversification recommended")
        sys.exit()

    portfolio = data[scaled.keys()]
    returns = np.log(portfolio) - np.log(portfolio.shift(1))
    returns = returns.iloc[1:, :]  # Remove NaN row after log
    weights_vector = list(scaled.values())

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

    sorted_weights = sorted(scaled.items(), key=lambda kv: kv[1], reverse=config.sort_by_weights)

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
