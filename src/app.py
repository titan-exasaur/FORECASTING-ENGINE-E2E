import pandas as pd
import streamlit as st
from forecasting_engine.data.ingestion import *
from forecasting_engine.data.cleansing import *

# -----------------------------------
# 0. SETTING UP APPLICATION BASE
# -----------------------------------
st.title("📈 FORECASTING ENGINE")


# -----------------------------------
# 1. DATA INGESTION
# -----------------------------------

file = st.file_uploader(
    "Upload a file for demand forecasting",
    type=["csv", "parquet", "xls", "xlsx"]
)
if file:
    raw_df = data_loader(file)

    st.header("DATA PREVIEW")
    st.dataframe(raw_df)

    st.header("DATA MAPPING")
    map_dict = data_columns_mapper(raw_df)
    st.write(map_dict)

    st.header("DATA CLEANSING")
    cleansed_df = data_cleanser(raw_df,
                                datetime_col=map_dict['datetime_col'],
                                demand_col=map_dict['demand_col'])
    data_continuity = check_data_continuity(cleansed_df=cleansed_df, 
                                            datetime_col=map_dict['datetime_col'],
                                            frequency=map_dict['frequency'])
    
    st.info(f' Data continuity before imputation: {data_continuity}')
    imputed_data = data_imputer(cleansed_df=cleansed_df,
                                datetime_col=map_dict['datetime_col'],
                                demand_col=map_dict['demand_col'],
                                frequency=map_dict['frequency'],
                                data_continuity=data_continuity)
    
    data_continuity_after = check_data_continuity(
        cleansed_df=imputed_data,
        datetime_col=map_dict['datetime_col'],
        frequency=map_dict['frequency']
    )
    st.info(f' Data continuity after imputation: {data_continuity_after}')


else:
    st.error("Upload data")
