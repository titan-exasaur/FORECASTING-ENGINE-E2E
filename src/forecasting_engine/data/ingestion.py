import os
import json
import pandas as pd
import streamlit as st
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv
from forecasting_engine.logger import app_logger 

load_dotenv()

logger = app_logger(__name__)

CONFIG_PATH = Path(os.getenv("CONFIG_PATH"))


def data_loader(file) -> pd.DataFrame:
    if file is None:
        return None

    st.success("File upload successful")
    logger.info("File upload successful")

    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".parquet"):
        return pd.read_parquet(file)
    elif file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        logger.error("Unsupported file format")
        return None


def data_columns_mapper(raw_df: pd.DataFrame) -> Dict:
    """
    Maps the columns from the uploaded data to the schema
    """

    df_cols = list(raw_df.columns)

    datetime_col = st.selectbox(
        "Choose the datetime column",
        options=df_cols
    )

    DATETIME_FORMATS = [
    "YYYY-MM-DD",
    "DD-MM-YYYY",
    "MM-DD-YYYY",
    "YYYY/MM/DD",
    "DD/MM/YYYY",
    "MM/DD/YYYY",

    "YYYY-MM-DD HH:MM",
    "YYYY-MM-DD HH:MM:SS",
    "DD-MM-YYYY HH:MM",
    "DD-MM-YYYY HH:MM:SS",
    "MM/DD/YYYY HH:MM",
    "MM/DD/YYYY HH:MM:SS",

    "ISO 8601 (YYYY-MM-DDTHH:MM:SS)",
    "ISO 8601 with TZ (YYYY-MM-DDTHH:MM:SSZ)",

    "YYYYMMDD",
    "YYYYMMDDHH"]

    datetime_format = st.selectbox(
        label="Select format of the datetime column",
        options=DATETIME_FORMATS
    )

    remaining_cols = [c for c in df_cols if c != datetime_col]

    demand_col = st.selectbox(
        "Choose the demand column",
        options=remaining_cols
    )

    frequency = st.selectbox("Select frequency of your data",
                             options=['hourly', 'daily',
                                      'weekly', 'monthly', 'quarterly', 
                                      'annual'])

    map_dict = {
        "datetime_col": datetime_col,
        "datetime_format": datetime_format,
        'frequency': frequency,
        "demand_col": demand_col
    }

    logger.info('Data Mapping Complete')

    with open(CONFIG_PATH/"data_mapping.json", "w") as f:
        json.dump(map_dict, f, indent=4)

    logger.info("Data mapping saved")

    return map_dict
