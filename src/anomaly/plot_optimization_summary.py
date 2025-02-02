from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_optimization_summary(optimization_summary: List[Dict[str, float]]):
    """
    Visualizes the optimization summary using a bar and line chart.
    """
    df = pd.DataFrame(optimization_summary)
    df.sort_values("threshold", inplace=True, ascending=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Optimal Thresholds per Stock", "Scores & Anomaly Fraction"),
        specs=[[{}], [{"secondary_y": True}]],
    )

    fig.add_trace(go.Bar(x=df["stock"], y=df["threshold"], name="Threshold"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["stock"], y=df["best_score"], mode="lines+markers", name="Best Score"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["stock"], y=df["anomaly_fraction"], mode="lines+markers", name="Anomaly Fraction"), row=2, col=1, secondary_y=True)

    fig.update_xaxes(title_text="Stock Ticker", row=2, col=1)
    fig.update_yaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Score", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Anomaly Fraction", secondary_y=True, row=2, col=1)

    fig.update_layout(
        title="Optimization Summary",
        height=800,
        legend=dict(yanchor="top", y=1.05, xanchor="left", x=0),
    )

    fig.show()
