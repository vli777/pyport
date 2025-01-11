import pandas as pd
import plotly.graph_objects as go
import numpy as np

from config import Config


def plot_graphs(
    daily_returns, cumulative_returns, config: Config, symbols, bgcolor="#f4f4f4"
):
    """
    Creates Plotly graphs for daily and cumulative returns with specified customizations.

    Parameters:
    - daily_returns (pd.DataFrame): DataFrame containing daily returns for each symbol.
    - cumulative_returns (pd.DataFrame): DataFrame containing cumulative returns for each symbol.
    - config (object): Configuration object with boolean attributes:
        - plot_daily_returns (bool)
        - plot_cumulative_returns (bool)
        - sort_by_weights (bool)
    - symbols (list): List of symbols corresponding to columns in `cumulative_returns`
    - bgcolor (str): Background color for the plots.
    """

    # Helper to generate a unified color mapping
    def generate_color_map(symbols, cumulative_returns):
        """
        Generates a color map from red to violet based on sorted cumulative returns.

        Parameters:
        - symbols (list): List of symbols
        - cumulative_returns (pd.DataFrame): DataFrame containing cumulative returns.

        Returns:
        - dict: Mapping from symbol to color.
        """
        # Verify that all symbols are present in cumulative_returns
        missing_symbols = [
            symbol for symbol in symbols if symbol not in cumulative_returns.columns
        ]
        if missing_symbols:
            print(
                f"Warning: The following symbols are missing in cumulative_returns and will be skipped: {missing_symbols}"
            )
            # Exclude missing symbols
            symbols = [
                symbol for symbol in symbols if symbol in cumulative_returns.columns
            ]

        if not symbols:
            raise ValueError("No valid symbols provided for plotting.")

        # Sort symbols based on their final cumulative return value (ascending)
        sorted_symbols = sorted(symbols, key=lambda x: cumulative_returns[x].iloc[-1])

        num_symbols = len(sorted_symbols)
        if num_symbols > 1:
            # Generate hues from 0 (red) to 270 (violet)
            hues = np.linspace(0, 270, num_symbols, endpoint=False)
        elif num_symbols == 1:
            hues = [0]  # Assign red if only one symbol

        colors = ["hsl({}, 70%, 50%)".format(int(h)) for h in hues]

        # Create color map
        color_map = {symbol: color for symbol, color in zip(sorted_symbols, colors)}

        # Assign gold color to SIM_PORT
        color_map["SIM_PORT"] = "hsl(50, 100%, 50%)"  # Gold

        return color_map, sorted_symbols

    # Generate color map and get sorted symbols
    color_map, sorted_symbols = generate_color_map(symbols, cumulative_returns)

    # Helper to update the layout for both plots
    def update_plot_layout(
        fig, title=None, hovermode="x unified", cumulative_returns=None
    ):
        fig.update_layout(
            hoverlabel=dict(
                font=dict(size=16), namelength=-1
            ),  # Hide the trace name in hover
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            title=title if title else "",
            hovermode=hovermode,  # Set hover mode
            xaxis=dict(
                showgrid=False,  # Disable gridlines for x-axis
                zeroline=False,  # Don't show the x-axis zero line
                showticklabels=True,  # Ensure x-axis tick labels (dates) are shown
                tickmode="auto",  # Automatically space the ticks
                tickformat="%b %Y",  # Set the tick format to Month/Day/Year
                ticks="outside",  # Optional: draw ticks outside of the plot
                type="date",  # Ensure the x-axis is treated as date type
            ),
            yaxis=dict(
                showgrid=True,  # Enable gridlines for y-axis
                zeroline=False,  # Don't show the y-axis zero line
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            hoverdistance=10,
        )

    # Function to generate hovertemplate with symbol name and difference for cumulative returns
    def get_hovertemplate_with_diff():
        hovertemplate = (
            "%{meta}: %{y:.2%}"  # Use %{meta} for symbol name
            ' (<span style="color:%{customdata[1]}">%{customdata[0]:+.2%}</span>)'  # Colored difference
            "<extra></extra>"  # Hide the trace name in hover box
        )
        return hovertemplate

    # Function to generate regular hovertemplate (for cumulative returns without difference)
    def get_hovertemplate():
        return "%{meta}: %{y:.2%}<extra></extra>"  # Use %{meta} for symbol name

    # Plot daily returns with custom hover template
    def plot_daily_returns():
        if config.plot_daily_returns:
            fig2 = go.Figure(
                [
                    go.Box(y=daily_returns[col], marker_color=color_map[col], name=col)
                    for i, col in enumerate(daily_returns.columns)
                ]
            )
            fig2.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(zeroline=False, gridcolor="white"),
                paper_bgcolor=bgcolor,
                plot_bgcolor=bgcolor,
                title="Daily Returns",
            )
            fig2.show()

    # Plot cumulative returns with custom hover template
    def plot_cumulative_returns():
        if config.plot_cumulative_returns:
            fig = go.Figure()

            # Get SIM_PORT data if available
            sim_port_data = cumulative_returns.get("SIM_PORT", None)

            # Sort by weights if required
            if config.sort_by_weights:
                sorted_cols = cumulative_returns.sort_values(
                    cumulative_returns.index[-1], ascending=False, axis=1
                ).columns
                cumulative_returns_sorted = cumulative_returns[sorted_cols]
            else:
                cumulative_returns_sorted = cumulative_returns

            # Add traces for each column
            for i, col in enumerate(cumulative_returns_sorted.columns):
                is_sim_port = col == "SIM_PORT"
                col_data = cumulative_returns_sorted[col]

                if not is_sim_port and sim_port_data is not None:
                    # Compute the difference from SIM_PORT
                    diff = col_data - sim_port_data
                    # Determine color based on sign of difference
                    color = ["green" if d >= 0 else "red" for d in diff]
                    # Prepare customdata with difference and color
                    customdata = np.array([diff.values, color], dtype=object).T

                    fig.add_trace(
                        go.Scatter(
                            x=cumulative_returns_sorted.index,
                            y=col_data,
                            mode="lines",
                            meta=col,
                            name=col,
                            line=dict(width=2, color=color_map[col]),
                            opacity=0.7,
                            hovertemplate=get_hovertemplate_with_diff(),
                            customdata=customdata,  # Store the difference and color
                        )
                    )
                else:
                    # SIM_PORT trace
                    # Convert datetime index to numerical values (days since the first date)
                    time_values = (
                        cumulative_returns_sorted.index
                        - cumulative_returns_sorted.index[0]
                    ).days

                    # Normalize the time values between 0 and 1 for the gradient
                    normalized_time_values = (time_values - np.min(time_values)) / (
                        np.max(time_values) - np.min(time_values)
                    )

                    # Create line segments to approximate the gradient effect
                    x_values = cumulative_returns_sorted.index
                    y_values = cumulative_returns_sorted["SIM_PORT"]

                    # Add markers and hover info for the full line (with a single color)
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode="markers+lines",
                            name="SIM_PORT",
                            line=dict(width=3, color="gold"),  # SIM_PORT line in gold
                            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
                            showlegend=True,  # Enable legend for SIM_PORT
                        )
                    )

                    # Plot each segment with a corresponding color
                    # for i in range(1, len(x_values)):
                    #     fig.add_trace(go.Scatter(
                    #         x=[x_values[i-1], x_values[i]],  # Each segment has two x points
                    #         y=[y_values[i-1], y_values[i]],  # Each segment has two y points
                    #         mode='lines',
                    #         line=dict(color=f'rgb({255 - int(255 * normalized_time_values[i])}, {int(165 * normalized_time_values[i])}, 0)', width=3),
                    #         showlegend=False,  # Disable legend for individual segments
                    #         hoverinfo='skip'  # Disable hover info for individual segments
                    #     ))

                    # Update layout
                    fig.update_layout(
                        title="Cumulative Returns",
                        # xaxis_title="Date",
                        # yaxis_title="Cumulative Returns",
                        paper_bgcolor=bgcolor,
                        plot_bgcolor=bgcolor,
                    )

            # Display the figure
            fig.show()

    # Generate plots
    plot_daily_returns()
    plot_cumulative_returns()
