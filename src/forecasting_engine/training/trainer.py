import numpy as np
import pandas as pd
from forecasting_engine.training.evaluator import rmse
from forecasting_engine.models.sarimax_model import SARIMAXModel
from forecasting_engine.training.splitter import time_series_split

def model_trainer(preprocessed_df: pd.DataFrame,
                  model_config: dict,
                  map_dict: dict):

    y = preprocessed_df[map_dict['demand_col']]
    n_splits = model_config["splitting"]["n_splits"]
    model_params = model_config["model"]["params"]

    best_score = float("inf")
    best_model = None
    best_y_test = None
    best_preds = None

    for fold, (train_idx, test_idx) in enumerate(
        time_series_split(y=y, n_splits=n_splits), start=1
    ):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = SARIMAXModel(
            order=tuple(model_params["order"]),
            seasonal_order=tuple(model_params["seasonal_order"])
        )

        trained_model = model.fit(y_train)
        preds = trained_model.predict(steps=len(y_test))

        score = rmse(y_test.values, preds.values)

        if score < best_score:
            best_score = score
            best_model = trained_model
            best_y_test = y_test
            best_preds = preds

    return best_model, best_y_test, best_preds, best_score
