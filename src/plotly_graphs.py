import plotly.graph_objects as go
import numpy as np

def plot_graphs(daily_returns, cumulative_returns, avg, config, symbols, bgcolor="#f4f4f4"):
    """
    Creates Plotly graphs for daily and cumulative returns with specified customizations.

    Parameters:
    - daily_returns (pd.DataFrame): DataFrame containing daily returns for each symbol.
    - cumulative_returns (pd.DataFrame): DataFrame containing cumulative returns for each symbol.
    - avg (set): Set of symbols that should have higher opacity.
    - config (object): Configuration object with boolean attributes:
        - plot_daily_returns (bool)
        - plot_cumulative_returns (bool)
        - sort_by_weights (bool)
    - symbols (list): List of symbols corresponding to columns in `cumulative_returns` excluding 'SIM_PORT'.
    - bgcolor (str): Background color for the plots.
    """

    # Helper to generate colors for plotting
    def generate_colors(num_items):
        return ["hsl(" + str(h) + ",50%,50%)" for h in np.linspace(0, 360, num_items, endpoint=False)]

   # Identify all unique symbols (including 'SIM_PORT' if present)
    unique_symbols = symbols.copy()
    if "SIM_PORT" in cumulative_returns.columns:
        unique_symbols.append("SIM_PORT")

    # Generate a single color list based on the number of unique symbols
    colors = generate_colors(len(unique_symbols))
    symbol_color_map = {symbol: color for symbol, color in zip(unique_symbols, colors)}
    
    # Helper to update the layout for both plots
    def update_plot_layout(fig, title=None, hovermode='x unified'):
        fig.update_layout(
            hoverlabel=dict(font=dict(size=14), namelength=-1),  # Hide the trace name in hover
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            title=title if title else "",
            hovermode=hovermode,  # Set hover mode
            xaxis=dict(showgrid=False, zeroline=False, type='category', showticklabels=True),
            yaxis=dict(showgrid=True, zeroline=False),
            margin=dict(l=40, r=40, t=40, b=40),
            hoverdistance=10,
        )

    # Function to generate hovertemplate with symbol name and difference for cumulative returns
    def get_hovertemplate_with_diff():
        hovertemplate = (
            '%{meta}: %{y:.2%}'  # Use %{meta} for symbol name
            ' (<span style="color:%{customdata[1]}">%{customdata[0]:+.2%}</span>)'  # Colored difference
            '<extra></extra>'  # Hide the trace name in hover box
        )
        return hovertemplate

    # Function to generate regular hovertemplate (for cumulative returns without difference)
    def get_hovertemplate():
        return '%{meta}: %{y:.2%}<extra></extra>'  # Use %{meta} for symbol name

    # Plot daily returns with custom hover template
    def plot_daily_returns():
        if config.plot_daily_returns:
            colors = generate_colors(len(daily_returns.columns))
            fig2 = go.Figure([go.Box(y=daily_returns[col], marker_color=colors[i], name=col)
                          for i, col in enumerate(daily_returns.columns)])
            fig2.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(zeroline=False, gridcolor="white"),
                paper_bgcolor=bgcolor,
                plot_bgcolor=bgcolor,
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
                    color = ['green' if d >= 0 else 'red' for d in diff]
                    # Prepare customdata with difference and color
                    customdata = np.array([diff.values, color], dtype=object).T

                    fig.add_trace(go.Scatter(
                        y=col_data,
                        mode="lines",
                        meta=col,  
                        name=col,
                        line=dict(width=2, color=symbol_color_map[col]),
                        opacity=1.0, #if is_sim_port else 0.5,
                        hovertemplate=get_hovertemplate_with_diff(),
                        customdata=customdata  # Store the difference and color
                    ))
                else:
                    # SIM_PORT trace
                    fig.add_trace(go.Scatter(
                        y=col_data,
                        mode="lines+markers",
                        meta=col,  # Use column name for SIM_PORT
                        name=col,
                        line=dict(width=3, color=symbol_color_map[col]),
                        opacity=1.0,
                        hovertemplate=get_hovertemplate(),  # Use regular hovertemplate
                        showlegend=True  # Show in legend
                    ))

            # Apply layout settings with unified hover mode
            update_plot_layout(fig, "Cumulative Returns", hovermode='x unified')
            # Display the figure
            fig.show()

    # Generate plots
    plot_daily_returns()
    plot_cumulative_returns()
