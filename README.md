# 📈 Forecasting Engine

## Objective

Build a **general-purpose time series forecasting engine** that can
ingest user data and automatically produce reliable forecasts with
visualization and evaluation.

The engine is designed to: - Accept **any dataset** with a datetime
column and demand/target column - Handle multiple **frequencies**
(hourly, daily, weekly, monthly, quarterly) - Perform robust **data
cleansing, continuity checks, and imputation** - Train multiple
forecasting models and automatically select the **best-performing
model** - Generate **interactive visualizations** and evaluation metrics

------------------------------------------------------------------------

## Key Inputs

-   Datetime column (user-mapped)
-   Datetime format (user-selected)
-   Demand / target column
-   Data frequency
-   Forecast window size

------------------------------------------------------------------------

## Data Processing Pipeline

1.  **Data Ingestion**
    -   CSV / Excel / Parquet upload via Streamlit
2.  **Column Mapping**
    -   User maps datetime and demand columns
    -   Frequency selection
3.  **Data Cleansing**
    -   Datetime coercion
    -   NaN / NaT handling
    -   Duplicate removal
    -   Negative demand handling
4.  **Continuity Check**
    -   Detects missing timestamps based on frequency
5.  **Temporal Imputation**
    -   Reindexing to a complete datetime grid
    -   Missing demand filled (default: zero-fill)
6.  **Optional Enhancements (Planned)**
    -   Winsorization
    -   ACT-PCT scaling
    -   Advanced imputation strategies

------------------------------------------------------------------------

## Modeling Strategy

-   Train **3 different forecasting models**
-   Evaluate using:
    -   RMSE
    -   MAE
    -   WMAPE
-   Automatically select the **best model**
-   Generate final forecast for the requested horizon

------------------------------------------------------------------------

## Visualization

-   Interactive **Plotly** charts
-   Actual vs Forecast comparison
-   Clean, production-ready UI via Streamlit

------------------------------------------------------------------------

## Current Status

✅ Data ingestion and mapping\
✅ Datetime parsing and frequency handling\
✅ Data cleansing and validation\
✅ Continuity checks and imputation\
🚧 Model training and selection (in progress)\
🚧 Forecast evaluation metrics (in progress)\
🚧 Plotly visualization (in progress)

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Pandas, NumPy
-   Statsmodels / ML models (planned)
-   Plotly
-   Streamlit
-   Docker (deployment planned)
-   AWS (EC2 + S3 planned)

------------------------------------------------------------------------

## Vision

A **plug-and-play forecasting engine** usable across domains with
minimal configuration and strong statistical guarantees.
