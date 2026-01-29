import os
import pandas as pd
import streamlit as st
from datetime import datetime
from forecasting_engine.utils import *
from forecasting_engine.logger import app_logger
from forecasting_engine.data.ingestion import *
from forecasting_engine.data.cleansing import *
from forecasting_engine.data.preprocessing import *
from forecasting_engine.training.trainer import model_trainer
from forecasting_engine.training.evaluator import model_evaluator

# -----------------------------------
# 0. SETTING UP APPLICATION BASE
# -----------------------------------
st.title("📈 FORECASTING ENGINE")

if "RUN_ID" not in os.environ:
    os.environ["RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logger = app_logger(__name__)


# -----------------------------------
# 1. DATA INGESTION
# -----------------------------------

file = st.file_uploader(
    "Upload a file for demand forecasting",
    type=["csv", "parquet", "xls", "xlsx"]
)
if file:
    raw_df = data_loader(file)
    raw_data_saver(raw_df)

    st.header("DATA PREVIEW")
    st.dataframe(raw_df)

    st.header("DATA MAPPING")
    map_dict = data_columns_mapper(raw_df)
    # st.write(map_dict)

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
    best_model, y_test, preds, score = model_trainer(
        preprocessed_df=preprocessed_df,
        model_config=model_config,
        map_dict=map_dict
    )

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Forecast": preds.values
    })
    st.dataframe(results_df)


    st.subheader("📊 Actual vs Forecast")
    plot_df = results_df.copy()
    plot_df = preprocessed_df.loc[y_test.index, [
        map_dict['datetime_col']
    ]].copy()

    plot_df["Actual"] = y_test.values
    plot_df["Forecast"] = preds.values
    fig = plot_actual_vs_forecast(
        df=plot_df,
        datetime_col=map_dict['datetime_col'],
        actual_col="Actual",
        forecast_col="Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)



    st.subheader("MODEL EVALUATION")
    mae, rmse, wmape = model_evaluator(y_true=results_df['Actual'],
                                         y_pred=results_df['Forecast'])
    st.error(f"MEAN ABSOLUTE ERROR: {round(mae, 2)}")
    st.info(f"ROOT MEAN SQUARED ERROR: {round(rmse, 2)}")
    st.warning(f"WEIGHTED MEAN ABSOLUTE PERCENTAGE ERROR: {round(wmape, 2)}%")


    

else:
    st.error("Upload data")
