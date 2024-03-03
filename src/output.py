from mlfinlab.backtest_statistics import sharpe_ratio
import plotly.graph_objects as go
import numpy as np
import csv
import os

from portfolio import *

def save_model_results(model_name, time_period, input_filename, symbols, scaled):
    CWD = os.getcwd() + "/"
    if not os.path.exists(CWD + "cache"):
        os.makedirs(CWD + "cache")
    output_file = (
        CWD + "cache/" + f"{input_filename}-{model_name}-{time_period}.csv")
    print ('saving results to model cache')
    writer = csv.writer(open(output_file, "w", newline=""))
    for key, val in scaled.items():
        writer.writerow([key, val])
    filtered_symbols = [sym for sym in symbols if sym not in scaled.keys()]
    for symbol in filtered_symbols:
        writer.writerow([symbol, 0])

def output(
    data,
    allocation_weights,
    inputs,
    start_date,
    end_date,
    symbols,
    stack,
    max_size=21,    
    sort_by_weights=False,
    optimization_model=None,
    time_period=1.0,
    minimum_weight=0.01,
    test_mode=False,    
):
    """
    produces console output with descriptive statistics of the provided portfolio over a date range
    """
    if isinstance(allocation_weights, dict):
        clean_weights = allocation_weights
    else:
        clean_weights = allocation_weights.to_dict("records")[0]
    if test_mode:
        print("raw weights", clean_weights)

    scaled = scale_to_one(clean_weights)

    while len(scaled) > 0 and min(scaled.values()) < minimum_weight or len(scaled) > max_size:
        clipped = clip_by_weight(scaled, minimum_weight)
        scaled = scale_to_one(clipped)
        minimum_weight = get_min_by_size(scaled, max_size, minimum_weight)

    stack[optimization_model + str(time_period)] = [scaled, len(scaled)]
    save_model_results(model_name=optimization_model, time_period=time_period, input_filename=inputs, symbols=symbols, scaled=scaled)

    if len(scaled) > 0:
        portfolio = data[scaled.keys()]
        returns = np.log(portfolio) - np.log(portfolio.shift(1))
        returns = returns.iloc[1:, :]
        weights_vector = list(scaled.values())

        weighted_returns = returns.mul(weights_vector)   

        portfolio_returns = weighted_returns.apply(np.sum, axis=1)      
        portfolio_cumulative_returns = portfolio_returns.add(1).cumprod()        

        try:
            sharpe = sharpe_ratio(portfolio_returns)    
        except:
            sharpe = 0

        print(f"\ntime period: {start_date} to {end_date} ({time_period} yrs)")
        print("inputs:", inputs)
        print("optimization methods:", optimization_model)
        print("sharpe ratio:", round(sharpe, 2))
        print(f"cumulative return: { round((portfolio_cumulative_returns[-1] - 1) * 100, 2)}%")
        print(
            f"portfolio allocation weights (min {minimum_weight:.2f}):")
    else:
        print("max diversification recommended")

    if sort_by_weights:
        for symbol, weight in sorted(scaled.items(),
                                     key=lambda kv: (kv[1], kv[0]),
                                     reverse=True):
            print(symbol, f"\t{weight:.3f}")
    else:
        for symbol, weight in sorted(scaled.items()):
            print(symbol, f"\t{weight:.3f}")

    portfolio_returns = portfolio_returns.to_frame()
    portfolio_returns.columns=['SIM_PORT']

    portfolio_cumulative_returns = portfolio_cumulative_returns.to_frame()
    portfolio_cumulative_returns.columns=['SIM_PORT']

    all_daily_returns = returns.join(portfolio_returns)
    all_cumulative_returns = ((portfolio_cumulative_returns) - 1).join(returns.add(1).cumprod() - 1)    
    
    return all_daily_returns, all_cumulative_returns

def plot_graphs(daily_returns, cumulative_returns, avg, config):
    """
    creates plotly graphs
    """
    if config["test_mode"]:
        print(daily_returns, cumulative_returns)
    if config["plot_cumulative_returns"] or config["plot_daily_returns"]:
        # daily_returns = data.pct_change()[1:]
        if config["test_mode"]:
            print(daily_returns.head())

        # cumulative_returns = daily_returns.add(1).cumprod().sub(1).mul(100)
        if config["test_mode"]:
            print(cumulative_returns.head())

        if config["plot_cumulative_returns"]:
            if config["sort_by_weights"]:
                sorted_cols = cumulative_returns.sort_values(
                    cumulative_returns.index[-1], ascending=False,
                    axis=1).columns
                cumulative_returns = cumulative_returns[sorted_cols]

                fig = go.Figure()

                for col in cumulative_returns.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns[col],
                            mode="lines" if col != "SIM_PORT" else "lines+markers",
                            name=col,
                            line=dict(width=3 if col in avg.keys() else 2),
                            opacity=1 if col in avg.keys() else 0.6,
                        ))

            fig.show()

        if config["plot_daily_returns"]:
            colors = [
                "hsl(" + str(h) + ",50%" + ",50%)"
                for h in np.linspace(0, 360, len(daily_returns.columns))
            ]

            fig2 = go.Figure(data=[
                go.Box(y=daily_returns[col], marker_color=colors[i], name=col)
                for i, col in enumerate(daily_returns.columns)
            ])

            fig2.update_layout(
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(zeroline=False, gridcolor="white"),
                paper_bgcolor="rgb(233,233,233)",
                plot_bgcolor="rgb(233,233,233)",
            )

            fig2.show()
