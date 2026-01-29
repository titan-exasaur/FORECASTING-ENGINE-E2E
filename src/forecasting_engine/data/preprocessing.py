import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller
from forecasting_engine.logger import app_logger 
from forecasting_engine.utils import load_config, processed_data_saver

load_dotenv()

logger = app_logger(__name__)

model_config_path = os.getenv("MODEL_CONFIG_PATH")
if not model_config_path:
    raise ValueError(
        "MODEL_CONFIG_PATH is not set. Check .env or environment variables."
    )

model_config = load_config(model_config_path)
params = model_config.get("preprocessing", {})

min_len = params.get("minimum_length", 20)
p_thresh = params.get("p_value_threshold", 0.05)
winsor_thresh = params.get("winsorising_threshold", 1.5)


def stationarity_check(cleansed_df: pd.DataFrame,
                       demand_col: str) -> bool:
    """
    Checks stationarity of demand values.

    Args:
        cleansed_df: cleaned dataframe
        demand_col: name of demand column

    Returns:
        bool: True if stationary, False if else
    """
    
    ts = cleansed_df[demand_col].dropna()
    if len(ts) < min_len:
        logger.warning("Time series too short for ADF test, skipping differencing")
        return True

    try:
        result = adfuller(ts)
    except ValueError as e:
        logger.warning(f"ADF test failed: {e}. Treating series as stationary.")
        return True

    logger.info(f"ADF Statistic: {result[0]}")
    logger.info(f"p-value: {result[1]}")

    if result[1] > p_thresh:
        logger.info("Series is non-stationary, differencing is needed")
        return False
    else:
        logger.info("Series is stationary, no differencing is needed")
        return True

def data_preprocessing(cleansed_df: pd.DataFrame,
                       demand_col: str) -> pd.DataFrame:
    """
    Data preprocessing module, winsorises data and applies differencing if necessary.

    Args:
        cleansed_df: cleansed dataframe
        demand_col: name of demand column

    Returns:
        preprocessed_df: processed dataframe

    Order: Raw Demand -> Outlier Handling(Winsorize/Cap) -> Stationarity Check(ADF) -> Differencing(if needed) -> Model

    """
    preprocessed_df = cleansed_df.copy()

    # --- Winsorization ---
    q1 = preprocessed_df[demand_col].quantile(0.25)
    q3 = preprocessed_df[demand_col].quantile(0.75)
    iqr = q3 - q1

    threshold = winsor_thresh
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    num_winsorized = (
        (preprocessed_df[demand_col] < lower_bound) |
        (preprocessed_df[demand_col] > upper_bound)
    ).sum()

    preprocessed_df[demand_col] = preprocessed_df[demand_col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    logger.info(
        f"Winsorized {num_winsorized} values "
        f"using bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    # --- Stationarity ---
    is_stationary = stationarity_check(preprocessed_df, demand_col)

    # --- Differencing ---
    order = 0  # default

    if not is_stationary:
        order = params.get("differencing_order", 1)
        preprocessed_df[f"{demand_col}_diff"] = (
            preprocessed_df[demand_col].diff(order)
        )
        preprocessed_df = preprocessed_df.dropna()
    else:
        preprocessed_df[f"{demand_col}_diff"] = preprocessed_df[demand_col]


    processed_data_saver(preprocessed_df)

    logger.info(
        f"Preprocessing summary | "
        f"winsor_threshold={winsor_thresh}, "
        f"stationary={is_stationary}, "
        f"differencing_order={order if not is_stationary else 0}, "
        f"rows_out={len(preprocessed_df)}"
    )

    logger.info(
        f"Quantiles | Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}"
    )


    return preprocessed_df


