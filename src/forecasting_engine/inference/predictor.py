import pandas as pd
from forecasting_engine.utils import model_loader

def generate_forecast_plot_df(
    plot_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    y_test_index: pd.Index,
    datetime_col: str,
    frequency: str,
    window_size: int
) -> pd.DataFrame:
    """
    Generates a combined dataframe of recent history + future forecasts.

    Args:
        plot_df: Historical dataframe with Actual & Forecast columns
        preprocessed_df: Full preprocessed dataframe
        y_test_index: Index corresponding to test window
        datetime_col: Datetime column name
        frequency: Time frequency (daily, weekly, etc.)
        window_size: Forecast horizon

    Returns:
        combined_df: DataFrame containing history + forecast horizon
    """

    model = model_loader()
    forecasts = model.predict(steps=window_size)

    last_date = preprocessed_df.loc[y_test_index, datetime_col].max()

    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "ME",
        "quarterly": "Q",
        "hourly": "H"
    }

    freq = freq_map.get(frequency)
    if not freq:
        raise ValueError(f"Unsupported frequency: {frequency}")

    future_dates = pd.date_range(
        start=last_date,
        periods=window_size + 1,
        freq=freq
    )[1:]

    future_df = pd.DataFrame({
        datetime_col: future_dates,
        "Actual": [None] * len(forecasts),
        "Forecast": forecasts
    })

    history_df = plot_df.tail(len(y_test_index)).copy()

    combined_df = pd.concat(
        [history_df, future_df],
        ignore_index=True
    )

    return combined_df
