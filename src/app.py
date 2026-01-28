import pandas as pd
import streamlit as st
from forecasting_engine.data.ingestion import *
from forecasting_engine.data.cleansing import *
from forecasting_engine.data.preprocessing import *
from forecasting_engine.models.sarimax_model import SARIMAXModel
from forecasting_engine.training.splitter import time_series_split

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

    st.header("DATA PREPROCESSING")
    preprocessed_df = data_preprocessing(cleansed_df=cleansed_df,
                                         demand_col=map_dict['demand_col'])
    st.dataframe(preprocessed_df)

    st.header("MODEL TRAINING")
    for train_idx, test_idx in time_series_split(
            y=preprocessed_df[map_dict['demand_col']],
            n_splits = model_config["splitting"]["n_splits"]
        ):
        
        y=preprocessed_df[map_dict['demand_col']]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model_params = model_config["model"]["params"]

        model = SARIMAXModel(
            order=tuple(model_params["order"]),
            seasonal_order=tuple(model_params["seasonal_order"])
        )

        model.fit(y_train)
        preds = model.predict(steps=len(y_test))
        st.write(preds)

    

else:
    st.error("Upload data")
