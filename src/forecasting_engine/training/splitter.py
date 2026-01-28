import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

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
