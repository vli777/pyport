from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_optimization_summary(optimization_summary: List[Dict[str, float]]):
    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(optimization_summary)

    # Sort by threshold (or whichever column you prefer)
    df.sort_values("threshold", inplace=True, ascending=True)

    # Create subplots: 2 rows, 1 column, shared x-axis, second row has a secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Optimal Threshold", "Scores & Anomaly Fraction"),
        specs=[[{}], [{"secondary_y": True}]],
    )

    # --- Row 1: Bar for Threshold ---
    fig.add_trace(
        go.Bar(x=df["stock"], y=df["threshold"], name="Threshold"), row=1, col=1
    )

    # --- Row 2: Lines for Scores & Anomaly Fraction ---
    # Best Composite Score (left axis)
    fig.add_trace(
        go.Scatter(
            x=df["stock"], y=df["best_score"], mode="lines+markers", name="Best Score"
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # Mean Score (left axis)
    fig.add_trace(
        go.Scatter(
            x=df["stock"], y=df["mean_score"], mode="lines+markers", name="Mean Score"
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # Anomaly Fraction (right axis)
    fig.add_trace(
        go.Scatter(
            x=df["stock"],
            y=df["anomaly_fraction"],
            mode="lines+markers",
            name="Anomaly Fraction",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Update axes
    fig.update_xaxes(title_text="Stock Ticker", row=2, col=1)
    fig.update_yaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Score", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Anomaly Fraction", secondary_y=True, row=2, col=1)

    # Overall layout
    fig.update_layout(
        title="Optimization Summary",
        height=800,
        legend=dict(yanchor="top", y=1.05, xanchor="left", x=0),
    )

    fig.show()
