import pandas as pd
from forecasting_engine.logger import app_logger
from sklearn.model_selection import TimeSeriesSplit

logger = app_logger(__file__)

def time_series_split(
    y: pd.Series,
    n_splits: int
):
    """
    Generator yielding train/test indices for time series CV.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in tscv.split(y):
        yield train_idx, test_idx

    logger.info("Data splitted into train-test partitions")