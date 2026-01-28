import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from forecasting_engine.models.base import BaseTimeSeriesModel

class SARIMAXModel(BaseTimeSeriesModel):

    def __init__(self, order, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None

    def fit(self, y: pd.Series):
        self.model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = self.model.fit(disp=False)
        return self

    def predict(self, steps: int):
        if self.model_fit is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        return self.model_fit.forecast(steps=steps)


    def summary(self):
        if self.model_fit is None:
            raise RuntimeError("Model must be fitted before calling summary()")
        return self.model_fit.summary()

