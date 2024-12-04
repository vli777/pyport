import numpy as np
import plotly.graph_objects as go

def generate_boxplot_data(data):
    # Calculate numerical values for boxplot
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    lower_whisker = np.min(data)
    upper_whisker = np.max(data)
    
    # Calculate outliers if any
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_fence or x > upper_fence]

    # Return all the numerical values
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers": outliers
    }

def plot_boxplot(data, boxplot_data):
    # Create a box plot using the values
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=data,
        q1=[boxplot_data['q1']],
        median=[boxplot_data['median']],
        q3=[boxplot_data['q3']],
        lowerfence=[boxplot_data['lower_whisker']],
        upperfence=[boxplot_data['upper_whisker']],
        boxpoints='outliers',
        outlierpoints=boxplot_data['outliers'],
        name='Data'
    ))

    fig.show()

# Example usage
data = np.random.normal(size=100)
boxplot_data = generate_boxplot_data(data)

# Print numerical values for the boxplot
print("Boxplot numerical values:", boxplot_data)

# Plot the boxplot
plot_boxplot(data, boxplot_data)
