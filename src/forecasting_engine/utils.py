import os
import yaml
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import plotly.graph_objects as go
from forecasting_engine.logger import app_logger

load_dotenv()

logger = app_logger(__name__)

DATA_PATH_ENV = os.getenv("DATA_PATH")
DATA_PATH = Path(DATA_PATH_ENV)

def load_config(yaml_path: str) -> dict:
    """
    Loads contents of yaml file

    Args:
        yaml_path: path to the yaml file

    Returns:
        config: parsed YAML contents
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as file:
        config = yaml.safe_load(file)
    
    logger.info(f"Loaded config from {yaml_path}")

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

def raw_data_saver(raw_df: pd.DataFrame) -> None:
    """
    Saves the user uploaded data as a CSV into data/raw using RUN_ID.
    """

    if raw_df is None or raw_df.empty:
        raise ValueError("Empty or invalid dataframe received")

    run_id = os.getenv("RUN_ID")
    if not run_id:
        raise EnvironmentError("RUN_ID not set in environment")

    data_path = os.getenv("DATA_PATH")
    if not data_path:
        raise EnvironmentError("DATA_PATH not set in environment")

    raw_dir = Path(data_path) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_data_path = raw_dir / f"{run_id}.csv"
    raw_df.to_csv(raw_data_path, index=False)

    logger.info(f"Raw data saved successfully at {raw_data_path}")


def processed_data_saver(processed_df: pd.DataFrame) -> None:
    """
    Saves the user uploaded data as a CSV into data/processed using RUN_ID.
    """

    if processed_df is None or processed_df.empty:
        raise ValueError("Empty or invalid dataframe received")

    run_id = os.getenv("RUN_ID")
    if not run_id:
        raise EnvironmentError("RUN_ID not set in environment")

    data_path = os.getenv("DATA_PATH")
    if not data_path:
        raise EnvironmentError("DATA_PATH not set in environment")

    processed_dir = Path(data_path) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_data_path = processed_dir / f"{run_id}.csv"
    processed_df.to_csv(processed_data_path, index=False)

    logger.info(f"Processed data saved successfully at {processed_data_path}")