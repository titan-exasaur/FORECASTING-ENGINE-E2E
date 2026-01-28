# 📈 Forecasting Engine

## Objective

Build a **general-purpose time series forecasting engine** that can ingest user data and automatically produce reliable forecasts with visualization and evaluation.

The engine is designed to:
- Accept **any dataset** with a datetime column and demand/target column
- Handle multiple **frequencies** (hourly, daily, weekly, monthly, quarterly, yearly)
- Perform robust **data cleansing, continuity checks, and imputation**
- Train multiple forecasting models and automatically select the **best-performing model**
- Generate **interactive visualizations** and evaluation metrics

---

## Key Inputs

- Datetime column (user-mapped)
- Datetime format (user-selected)
- Demand / target column
- Data frequency
- Forecast horizon

---

## Data Processing Pipeline

1. **Data Ingestion**
   - CSV / Excel / Parquet upload via Streamlit

2. **Column Mapping**
   - User maps datetime and demand columns
   - Frequency selection

3. **Data Cleansing**
   - Datetime coercion
   - NaN / NaT handling
   - Duplicate removal
   - Negative demand handling

4. **Continuity Check**
   - Detects missing timestamps based on frequency

5. **Temporal Imputation**
   - Reindexing to a complete datetime grid
   - Missing demand filled (default: zero-fill)

6. **Data Preprocessing**
   - Outlier handling (Winsorization / capping)
   - Stationarity check (ADF test)
   - Differencing (if required)

---

## Modeling Strategy

- Modular model architecture with a common base interface
- Initial models:
  - SARIMAX
- Planned:
  - Prophet
  - Gradient Boosting / ML models

### Model Evaluation
Models are evaluated using:
- RMSE
- MAE
- WMAPE

The best-performing model is automatically selected.

---

## Data Splitting Strategy

- Time-series–aware cross-validation
- Uses rolling splits (`TimeSeriesSplit`)
- Preserves temporal ordering

---

## Visualization

- Interactive **Plotly** charts
- Actual vs Forecast comparison
- Forecast confidence intervals (planned)
- Clean, production-ready UI via Streamlit

---

## Current Status

✅ Data ingestion and mapping  
✅ Datetime parsing and frequency handling  
✅ Data cleansing and validation  
✅ Continuity checks and imputation  
✅ Preprocessing (winsorization + differencing)  
🚧 Model training and selection  
🚧 Forecast evaluation metrics  
🚧 Plotly visualization  
🚧 Deployment (Docker + AWS)

---

## Tech Stack

- Python
- Pandas, NumPy
- Statsmodels
- Scikit-learn
- Plotly
- Streamlit
- Docker (planned)
- AWS EC2 + S3 (planned)

---

## Vision

A **plug-and-play forecasting engine** usable across domains with minimal configuration, strong statistical guarantees, and production-grade extensibility.
