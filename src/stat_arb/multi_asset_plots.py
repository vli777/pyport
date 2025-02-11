import plotly.graph_objects as go
import pandas as pd


def plot_multi_asset_signals(
    spread_series, signals, title="Multi-Asset Mean Reversion Trading Signals"
):
    """
    Plots the spread series and overlays buy/sell signals using Plotly.

    Args:
        spread_series (pd.Series): The spread time series.
        signals (pd.DataFrame): DataFrame with trading signals (Position column).
        title (str): Plot title.

    Returns:
        None (displays an interactive Plotly plot).
    """
    fig = go.Figure()

    # Add spread series (mean reversion indicator)
    fig.add_trace(
        go.Scatter(
            x=spread_series.index,
            y=spread_series.values,
            mode="lines",
            name="Spread (Z-Score)",
            line=dict(color="blue"),
        )
    )

    # Extract buy/sell signals
    buy_signals = signals[signals["Position"] == 1]
    sell_signals = signals[signals["Position"] == -1]

    # Add Buy signals
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=spread_series.loc[buy_signals.index],
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", color="green", size=10),
        )
    )

    # Add Sell signals
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=spread_series.loc[sell_signals.index],
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", color="red", size=10),
        )
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Spread (Z-Score)",
        template="plotly_white",
    )

    fig.show()


def plot_multi_asset_cumulative_returns(
    strategy_returns,
    benchmark_returns,
    title="Cumulative Returns: Mean Reversion Strategy vs. Baseline Allocation",
):
    """
    Plots cumulative returns for the multi-asset strategy vs. benchmark with improved readability.

    Args:
        strategy_returns (pd.Series): Cumulative returns of the strategy.
        benchmark_returns (pd.Series): Cumulative returns of the benchmark.
        title (str): Chart title.
    """
    fig = go.Figure()

    # Strategy plot (solid line)
    fig.add_trace(
        go.Scatter(
            x=strategy_returns.index,
            y=strategy_returns.values,
            mode="lines",
            name="Multi-Asset Strategy",
            line=dict(color="blue", width=2),
        )
    )

    # Benchmark plot (dashed line)
    fig.add_trace(
        go.Scatter(
            x=benchmark_returns.index,
            y=benchmark_returns.values,
            mode="lines",
            name="Benchmark",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0, y=1.1, orientation="h"),
        xaxis=dict(showgrid=True, tickangle=-45),
        yaxis=dict(showgrid=True, tickformat=".4f"),  # Limit decimals to 4 places
    )

    fig.show()
