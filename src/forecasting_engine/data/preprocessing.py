import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller
from forecasting_engine.logger import app_logger 
from forecasting_engine.utils import load_config

load_dotenv()

logger = app_logger(__file__)

DATA_PATH = Path(os.getenv("DATA_PATH"))
model_config_path = os.getenv("MODEL_CONFIG_PATH")
if not model_config_path:
    raise ValueError(
        "MODEL_CONFIG_PATH is not set. Check .env or environment variables."
    )

model_config = load_config(model_config_path)
params = model_config["params"]


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
    if len(ts) < params['minimum_length']:
        logger.warning("Time series too short for ADF test, skipping differencing")
        return True

    result = adfuller(ts)
    logger.info(f"ADF Statistic: {result[0]}")
    logger.info(f"p-value: {result[1]}")

    if result[1] > params['p_value_threshold']:
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

    threshold = params['winsorising_threshold']
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
    if not is_stationary:
        logger.info(f"Created column: {demand_col}_diff using first-order differencing")
        
        order = params.get("differencing_order", 1)
        
        preprocessed_df[f"{demand_col}_diff"] = (
            preprocessed_df[demand_col].diff(order)
        )
        preprocessed_df = preprocessed_df.dropna()
    else:
        preprocessed_df[f"{demand_col}_diff"] = preprocessed_df[demand_col]

    # saving processed data
    output_dir = DATA_PATH / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "preprocessed_data.csv"
    preprocessed_df.to_csv(output_path, index=False)

    logger.info(f"Preprocessed data saved to {output_path}")

    return preprocessed_df


