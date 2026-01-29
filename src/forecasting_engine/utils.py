import yaml
import pandas as pd
import plotly.graph_objects as go


def load_config(yaml_path: str) -> dict:
    """
    Loads contents of yaml file

    Args:
        yaml_path: path to the yaml file

    Returns:
        config: parsed YAML contents
    """
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    
    return config

def plot_actual_vs_forecast(
    df: pd.DataFrame,
    datetime_col: str,
    actual_col: str,
    forecast_col: str,
    title: str = "Actual vs Forecast"
):
    """
    Plots actual vs forecast values for time series data.

    Args:
        df: DataFrame containing datetime, actual, and forecast columns
        datetime_col: Name of datetime column
        actual_col: Column name for actual values
        forecast_col: Column name for forecasted values
        title: Plot title

    Returns:
        fig: Plotly Figure object
    """

    fig = go.Figure()

    # Actual values (RED)
    fig.add_trace(
        go.Scatter(
            x=df[datetime_col],
            y=df[actual_col],
            mode="lines+markers",
            name="Actual",
            line=dict(color="red", width=2)
        )
    )

    # Forecast values (BLUE)
    fig.add_trace(
        go.Scatter(
            x=df[datetime_col],
            y=df[forecast_col],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="blue", width=2, dash="dash")
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Demand",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
