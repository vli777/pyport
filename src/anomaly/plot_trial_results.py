import plotly.express as px
import pandas as pd
import optuna

def plot_trial_results(study: optuna.study.Study):
    # Collect data from each trial.
    data = []
    for trial in study.trials:
        if "mean_score" in trial.user_attrs and "std_score" in trial.user_attrs:
            data.append({
                "threshold": trial.params["threshold"],
                "mean_score": trial.user_attrs["mean_score"],
                "std_score": trial.user_attrs["std_score"],
                "anomaly_fraction": trial.user_attrs.get("anomaly_fraction")
            })
    if not data:
        print("No trial data to plot.")
        return
    df = pd.DataFrame(data)
    df = df.sort_values(by="threshold")
    # Create a bar chart with error bars representing the standard deviation.
    fig = px.bar(
        df,
        x="threshold",
        y="mean_score",
        error_y="std_score",
        hover_data=["anomaly_fraction"],
        title="Trial Results: Mean Score vs Threshold"
    )
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Mean Score")
    fig.show()
