import pandas as pd
from abc import ABC, abstractmethod
from statsmodels.tsa.statespace.sarimax import SARIMAX


class BaseTimeSeriesModel(ABC):
    @abstractmethod
    def fit(self, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, steps: int):
        pass

    @abstractmethod
    def summary(self):
        pass