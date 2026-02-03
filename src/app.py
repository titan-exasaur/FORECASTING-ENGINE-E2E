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
# UI CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Forecasting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    h1, h2, h3 { font-weight: 700; }
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc, #eef2f7);
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-weight: 600;
    }

    div[data-testid="stMetric"] div {
        color: #0f172a !important;
        font-weight: 700;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        padding: 20px;
        border-radius: 12px;
        border: 1px dashed #94a3b8;
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# 0. SETTING UP APPLICATION BASE
# -----------------------------------
st.title("📈 Forecasting Engine")
st.caption("Production-grade demand forecasting pipeline • Upload → Train → Evaluate → Forecast")

if "RUN_ID" not in os.environ:
    os.environ["RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logger = app_logger("forecasting_engine")
logger.info("Application started")

# -----------------------------------
# 1. DATA INGESTION
# -----------------------------------

with st.container():
    st.subheader("📂 Data Ingestion")
    file = st.file_uploader(
        "Upload a file for demand forecasting",
        type=["csv", "parquet", "xls", "xlsx"]
    )

if file:
    with st.spinner("Loading data..."):
        raw_df = data_loader(file)
        raw_data_saver(raw_df)

    st.success("Data loaded successfully")

    with st.container():
        st.subheader("🔍 Data Preview")
        st.dataframe(raw_df, use_container_width=True)

    with st.container():
        st.subheader("🗺️ Data Mapping")
        map_dict = data_columns_mapper(raw_df)

    with st.container():
        st.subheader("🧹 Data Cleansing")
        with st.spinner("Cleansing data..."):
            cleansed_df = data_cleanser(
                raw_df,
                datetime_col=map_dict['datetime_col'],
                demand_col=map_dict['demand_col']
            )

            data_continuity = check_data_continuity(
                cleansed_df=cleansed_df,
                datetime_col=map_dict['datetime_col'],
                frequency=map_dict['frequency']
            )

            st.info(f'📉 Data continuity before imputation: {data_continuity}')

            imputed_data = data_imputer(
                cleansed_df=cleansed_df,
                datetime_col=map_dict['datetime_col'],
                demand_col=map_dict['demand_col'],
                frequency=map_dict['frequency'],
                data_continuity=data_continuity
            )

            data_continuity_after = check_data_continuity(
                cleansed_df=imputed_data,
                datetime_col=map_dict['datetime_col'],
                frequency=map_dict['frequency']
            )

            st.success(f'📈 Data continuity after imputation: {data_continuity_after}')

    with st.container():
        st.subheader("⚙️ Data Preprocessing")
        with st.spinner("Preprocessing data..."):
            preprocessed_df = data_preprocessing(
                cleansed_df=cleansed_df,
                demand_col=map_dict['demand_col']
            )
        st.success("Data preprocessing complete")

    with st.container():
        st.subheader("🧠 Model Training")
        st.info("Training model... this may take a moment.")
        with st.spinner("Training in progress..."):
            best_model, y_test, preds, score = model_trainer(
                preprocessed_df=preprocessed_df,
                model_config=model_config,
                map_dict=map_dict
            )
            model_saver(best_model)

        st.success("Model training complete")

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Forecast": preds.values
    })

    plot_df = preprocessed_df.loc[y_test.index, [
        map_dict['datetime_col']
    ]].copy()

    plot_df["Actual"] = y_test.values
    plot_df["Forecast"] = preds.values

    with st.container():
        st.subheader("📊 Actual vs Forecast")
        fig = plot_actual_vs_forecast(
            df=plot_df,
            datetime_col=map_dict['datetime_col'],
            actual_col="Actual",
            forecast_col="Forecast"
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.subheader("🧪 Model Evaluation")
        mae, rmse, wmape = model_evaluator(
            y_true=results_df['Actual'],
            y_pred=results_df['Forecast']
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", round(mae, 2))
        col2.metric("RMSE", round(rmse, 2))
        col3.metric("WMAPE (%)", round(wmape, 2))

    with st.container():
        st.subheader("🔮 Model Inference")
        window_size = st.slider(
            "Select forecast window size",
            min_value=round(len(y_test) / 2),
            max_value=len(y_test),
            value=0
        )

        if window_size:
            with st.spinner("Generating forecast horizon..."):
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

            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload a dataset to begin forecasting")
