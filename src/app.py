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
from forecasting_engine.inference.predictor import generate_forecast_plot_df

# -----------------------------------
# 0. SETTING UP APPLICATION BASE
# -----------------------------------
st.title("📈 FORECASTING ENGINE")

if "RUN_ID" not in os.environ:
    os.environ["RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logger = app_logger("forecasting_engine")
logger.info("Application started")

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
    st.info("Data preprocessing complete")

    st.header("MODEL TRAINING")
    st.info("Model training started")
    best_model, y_test, preds, score = model_trainer(
        preprocessed_df=preprocessed_df,
        model_config=model_config,
        map_dict=map_dict
    )
    model_saver(best_model)

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Forecast": preds.values
    })
    st.success("Model training complete")


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

    st.plotly_chart(fig, width='stretch')



    st.subheader("MODEL EVALUATION")
    mae, rmse, wmape = model_evaluator(y_true=results_df['Actual'],
                                         y_pred=results_df['Forecast'])
    st.error(f"MEAN ABSOLUTE ERROR: {round(mae, 2)}")
    st.info(f"ROOT MEAN SQUARED ERROR: {round(rmse, 2)}")
    st.warning(f"WEIGHTED MEAN ABSOLUTE PERCENTAGE ERROR: {round(wmape, 2)}%")

    st.subheader("MODEL INFERENCE")
    window_size = st.slider("Enter window size: ", 
                                  min_value=round(len(y_test)/2),
                                  max_value=len(y_test),
                                  value=0)
    if window_size:
        combined_df = generate_forecast_plot_df(
            plot_df=plot_df,
            preprocessed_df=preprocessed_df,
            y_test_index=y_test.index,
            datetime_col=map_dict['datetime_col'],
            frequency=map_dict['frequency'],
            window_size=window_size
        )

        fig = plot_actual_vs_forecast(
            df=combined_df,
            datetime_col=map_dict['datetime_col'],
            actual_col="Actual",
            forecast_col="Forecast",
            title="Actuals + Forecast Horizon"
        )

        st.plotly_chart(fig, width='stretch')


    

else:
    st.error("Upload data")
