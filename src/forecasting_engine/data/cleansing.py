import pandas as pd
import streamlit as st

from forecasting_engine.logger import app_logger 
logger = app_logger(__file__)

def data_cleanser(raw_df: pd.DataFrame,
                   datetime_col: str,
                   demand_col: str) -> pd.DataFrame:
    """
    Cleanses the raw dataframe

    Args:
        raw_df: raw dataframe

    Returns:
        cleansed_df: cleansed dataframe
    """
    
    ts = raw_df.copy()

    ts[datetime_col] = pd.to_datetime(ts[datetime_col], errors="coerce")

    # dropping NaN / NaT values
    if ts[[datetime_col, demand_col]].isna().any().any():
        ts = ts.dropna(subset=[datetime_col, demand_col])
        st.warning("NaN values found, dropped successfully")

    # dropping duplicates
    if ts.duplicated().any():
        ts = ts.drop_duplicates()
        st.warning("Duplicate values found, dropped successfully")

    # handling negative demand
    if (ts[demand_col] < 0).any():
        st.warning("Negative values found in demand column, replacing with mean")
        mean_demand = ts.loc[ts[demand_col] >= 0, demand_col].mean()
        ts.loc[ts[demand_col] < 0, demand_col] = mean_demand

    return ts

def check_data_continuity(cleansed_df: pd.DataFrame,
                          datetime_col: str,
                          frequency: str) -> bool:
    """
    Checks if the datetime column is continuous

    Args:
        raw_df: Input raw dataframe
        frequency: frequency of data
    
    Returns:
        bool: True is continuous, False is not
    """

    ts = cleansed_df[datetime_col]
    ts = pd.to_datetime(ts, errors="coerce")
    ts = ts.dropna().sort_values().drop_duplicates()
    
    if len(ts) < 2:
        return True
    
    frequency_map = {
        "hourly": "H",
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q"
    }

    if frequency not in frequency_map:
        return False
    
    expected = pd.date_range(
        start=ts.min(),
        end=ts.max(),
        freq=frequency_map[frequency]
    )

    return ts.reset_index(drop=True).equals(
        pd.Series(expected)
    )

def data_imputer(cleansed_df: pd.DataFrame,
                 datetime_col: str,
                 demand_col: str,
                 frequency: str,
                 data_continuity: bool) -> pd.DataFrame:
    """
    Imputes the data for missing timestamp values to preserve temporal order

    Args:
        cleansed_df: Cleansed dataframe
        data_continuity: Boolean value -- result of check_data_continuity method

    Returns:
        imputed_df: Dataframe after imputation
    """

    if data_continuity:
        return cleansed_df

    freq_map = {
        "hourly": "H",
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q"
    }

    if frequency not in freq_map:
        return cleansed_df

    df = cleansed_df.copy()

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])

    df = df.set_index(datetime_col).sort_index()

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq_map[frequency]
    )

    df = df.reindex(full_index)

    # impute demand (time-series safe default)
    df[demand_col] = df[demand_col].fillna(0)

    df = df.reset_index().rename(columns={"index": datetime_col})

    return df
