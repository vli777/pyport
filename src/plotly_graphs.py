from pathlib import Path
from config import Config

import json
import colorsys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_colors_from_json(file_name: str, directory: str = "styles") -> dict:
    """
    Loads color palettes from a JSON file located in a specified directory.

    Parameters
    ----------
    file_name : str
        Name of the JSON file containing color definitions.
    directory : str, optional
        Directory where the JSON file is located, by default "styles".

    Returns
    -------
    dict
        A dictionary where keys are palette names and values are lists of hex color strings.
    """
    # Resolve the file path relative to the script location
    script_dir = Path(__file__).parent
    file_path = script_dir / directory / file_name

    # Load JSON
    with file_path.open("r") as file:
        return json.load(file)


def lighten_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Lightens a given hex color by blending it with white.

    Parameters
    ----------
    hex_color : str
        The original hex color (e.g., '#FF0000').
    factor : float, optional
        Amount to lighten the color, by default 0.3.

    Returns
    -------
    str
        A new, lightened color in hex format.
    """
    # Strip '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB (0-1 range)
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Convert to HLS, adjust lightness, and back to RGB
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1, l + factor)
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def generate_color_map(
    symbols: List[str],
    cumulative_returns: pd.DataFrame,
    palette: str = "default",
    color_dict: Dict[str, List[str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Generates a symbol-to-color mapping using either a custom palette
    or a default gradient. Symbols are sorted by their final cumulative
    return value (ascending).

    Parameters
    ----------
    symbols : List[str]
        List of symbols to plot.
    cumulative_returns : pd.DataFrame
        DataFrame containing cumulative returns, indexed by dates and
        columns as symbols.
    palette : str, optional
        Key for the color palette in `color_dict`, by default "default".
    color_dict : Dict[str, List[str]], optional
        A dictionary of color palettes, each containing a list of hex colors.
        By default None.

    Returns
    -------
    Tuple[Dict[str, str], List[str]]
        - A dict mapping each symbol to its assigned color.
        - A sorted list of valid symbols (in ascending order of final returns).
    """
    # Warn if symbols are missing
    missing_symbols = [s for s in symbols if s not in cumulative_returns.columns]
    if missing_symbols:
        warnings.warn(
            f"The following symbols are missing in cumulative_returns "
            f"and will be skipped: {missing_symbols}"
        )
        # Exclude missing symbols
        symbols = [s for s in symbols if s in cumulative_returns.columns]

    if not symbols:
        raise ValueError("No valid symbols provided for plotting.")

    # Sort symbols by final cumulative return (ascending)
    sorted_symbols = sorted(symbols, key=lambda x: cumulative_returns[x].iloc[-1])
    num_symbols = len(sorted_symbols)

    colors = []
    if color_dict and palette in color_dict:
        # Use custom colors from the specified palette
        base_colors = color_dict[palette]
        for i in range(num_symbols):
            # Cycle through the base colors
            base_color = base_colors[i % len(base_colors)]
            # Lighten further each time we wrap around the palette
            lightened_color = lighten_color(
                base_color, factor=0.2 * (i // len(base_colors))
            )
            colors.append(lightened_color)
    else:
        # Default color generation via hue sweep
        if num_symbols > 1:
            hues = np.linspace(0, 270, num_symbols, endpoint=False)
        else:
            # Only one symbol, default to red
            hues = [0]
        colors = [f"hsl({int(h)}, 70%, 50%)" for h in hues]

    # Build the color map
    color_map = {symbol: color for symbol, color in zip(sorted_symbols, colors)}

    # Force SIM_PORT to gold if present
    if "SIM_PORT" in color_map:
        color_map["SIM_PORT"] = "hsl(50, 100%, 50%)"

    return color_map, sorted_symbols


def update_plot_layout(
    fig: go.Figure,
    title: str = "",
    paper_bgcolor: str = "#f1f1f1",
    plot_bgcolor: str = "#0476D0",
    hovermode: str = "x unified",
) -> None:
    """
    Updates the layout for a Plotly figure with common styling options.

    Parameters
    ----------
    fig : go.Figure
        The figure to update.
    title : str, optional
        Plot title, by default "".
    paper_bgcolor : str, optional
        Background color of the paper, by default "#f1f1f1".
    plot_bgcolor : str, optional
        Background color of the plotting area, by default "#0476D0".
    hovermode : str, optional
        Hover behavior, by default "x unified".
    """
    fig.update_layout(
        title=title,
        hovermode=hovermode,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        hoverdistance=10,
        margin=dict(l=40, r=40, t=40, b=40),
        hoverlabel=dict(font=dict(size=16), namelength=-1),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickmode="auto",
            tickformat="%b %Y",
            ticks="outside",
            type="date",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
    )


def get_hovertemplate_with_diff() -> str:
    """
    Generates a hovertemplate that shows the symbol name, y-value as a percentage,
    and the difference from SIM_PORT in colored text.

    Returns
    -------
    str
        A Plotly hovertemplate string.
    """
    return (
        "%{meta}: %{y:.2%}"
        ' (<span style="color:%{customdata[1]}">%{customdata[0]:+.2%}</span>)'
        "<extra></extra>"
    )


def get_hovertemplate() -> str:
    """
    Generates a simpler hovertemplate that shows only the symbol name
    and y-value as a percentage.

    Returns
    -------
    str
        A Plotly hovertemplate string.
    """
    return "%{meta}: %{y:.2%}<extra></extra>"


def plot_daily_returns(
    daily_returns: pd.DataFrame,
    color_map: Dict[str, str],
    config: Config,
    paper_bgcolor: str,
    plot_bgcolor: str,
) -> None:
    """
    Plots daily returns as box plots for each symbol.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        DataFrame containing daily returns for each symbol.
    color_map : Dict[str, str]
        Mapping of symbols to colors.
    config : Config
        Configuration object controlling whether to plot daily returns.
    paper_bgcolor : str
        Background color for the figure.
    plot_bgcolor : str
        Background color for the plot area.
    """
    if not config.plot_daily_returns:
        return

    fig = go.Figure()
    for col in daily_returns.columns:
        fig.add_trace(
            go.Box(
                y=daily_returns[col], marker_color=color_map.get(col, "gray"), name=col
            )
        )

    fig.update_layout(
        title="Daily Returns",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
        yaxis=dict(showgrid=False, zeroline=False),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
    )

    fig.show()


def plot_cumulative_returns(
    cumulative_returns: pd.DataFrame,
    color_map: Dict[str, str],
    config: Config,
    paper_bgcolor: str,
    plot_bgcolor: str,
) -> None:
    """
    Plots cumulative returns as time-series lines, optionally showing difference
    vs. SIM_PORT in hover info.

    Parameters
    ----------
    cumulative_returns : pd.DataFrame
        DataFrame containing cumulative returns for each symbol.
    color_map : Dict[str, str]
        Mapping of symbols to colors.
    config : Config
        Configuration object controlling whether to plot cumulative returns and sorting.
    paper_bgcolor : str
        Background color for the figure.
    plot_bgcolor : str
        Background color for the plot area.
    """
    if not config.plot_cumulative_returns:
        return

    fig = go.Figure()

    # Retrieve SIM_PORT column if present
    sim_port_data = cumulative_returns.get("SIM_PORT")

    # Sort columns by final value if required

    # last_row = cumulative_returns.iloc[-1]  # final row (Series)
    # sorted_cols = last_row.sort_values(ascending=False).index
    # df_sorted = cumulative_returns[sorted_cols]

    df_sorted = cumulative_returns

    for col in df_sorted.columns:
        is_sim_port = col == "SIM_PORT"
        col_data = df_sorted[col]

        if not is_sim_port and sim_port_data is not None:
            # Show difference relative to SIM_PORT
            diff = col_data - sim_port_data
            diff_color = ["green" if d >= 0 else "red" for d in diff]
            customdata = np.array([diff.values, diff_color], dtype=object).T

            fig.add_trace(
                go.Scatter(
                    x=df_sorted.index,
                    y=col_data,
                    mode="lines",
                    meta=col,
                    name=col,
                    line=dict(width=3, color=color_map.get(col, "gray")),
                    hovertemplate=get_hovertemplate_with_diff(),
                    customdata=customdata,
                )
            )
        else:
            # For SIM_PORT or if SIM_PORT is not available
            fig.add_trace(
                go.Scatter(
                    x=df_sorted.index,
                    y=col_data,
                    mode="lines",
                    meta=col,
                    name=col,
                    line=dict(width=6, color=color_map.get(col, "gold")),
                    hovertemplate=get_hovertemplate(),
                )
            )

    update_plot_layout(
        fig,
        title="Cumulative Returns",
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
    )
    fig.show()


def plot_graphs(
    daily_returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    config: Config,
    symbols: List[str],
    paper_bgcolor: str = "#f4f4f4",
    plot_bgcolor: str = "#202346",
    palette: str = "default",
) -> None:
    """
    Creates Plotly graphs for daily returns and cumulative returns.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        DataFrame containing daily returns for each symbol.
    cumulative_returns : pd.DataFrame
        DataFrame containing cumulative returns for each symbol.
    config : Config
        Configuration object with boolean attributes:
            - plot_daily_returns (bool)
            - plot_cumulative_returns (bool)
            - sort_by_weights (bool)
    symbols : List[str]
        List of symbols corresponding to columns in `cumulative_returns`.
    paper_bgcolor : str, optional
        Background color for the entire figure (paper)
    plot_bgcolor : str, optional
        Background color for the plotting area
    """
    # Load color dictionary and generate color map
    color_dict = load_colors_from_json("color_themes.json")
    color_map, sorted_symbols = generate_color_map(
        symbols, cumulative_returns, palette=palette, color_dict=color_dict
    )

    # Plot daily returns
    plot_daily_returns(daily_returns, color_map, config, paper_bgcolor, plot_bgcolor)

    # Plot cumulative returns
    plot_cumulative_returns(
        cumulative_returns, color_map, config, paper_bgcolor, plot_bgcolor
    )
