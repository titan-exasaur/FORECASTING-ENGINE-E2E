import numpy as np
from forecasting_engine.logger import app_logger

logger = app_logger(__name__)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def wmape(y_true, y_pred):
    """
    Computes Weighted Mean Absolute Percentage Error (WMAPE).

    WMAPE measures forecast error weighted by actual demand, making it
    suitable for demand forecasting problems.

    Formula:
    WMAPE = ( Σ |y_true - y_pred| / Σ y_true ) × 100

    Notes:
    - Higher-demand periods contribute more to the final error
    - Scale-independent and easy to interpret as a percentage
    - Commonly used in forecasting benchmarks

    Edge Cases:
    - Returns NaN if total demand is zero
    - y_true and y_pred must have the same shape
    - Negative values in y_true are not allowed

    Args:
    - y_true: Actual observed values
    - y_pred: Model-predicted values

    Returns:
    - wmape: Percentage error (lower is better)
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if (y_true < 0).any():
        raise ValueError("Negative values in y_true not allowed for WMAPE")

    total = y_true.sum()
    if total == 0:
        return np.nan

    return np.sum(np.abs(y_true - y_pred)) / total * 100

def model_evaluator(y_true, y_pred):
    """
    Evaluates model predictions using standard regression metrics.

    Computes MAE, RMSE, and WMAPE to assess forecast accuracy.

    Args:
    - y_true: Actual observed values
    - y_pred: Model-predicted values

    Returns:
    - tuple: MAE, RMSE, and WMAPE scores
    """

    mae_score = mae(y_true, y_pred)
    rmse_score = rmse(y_true, y_pred)
    wmape_score = wmape(y_true, y_pred)

    eval_dict = {
        "mae": mae_score,
        "rmse": rmse_score,
        "wmape": wmape_score
    }

    logger.info(f"Model evaluation complete. Metrics: {eval_dict}")

    return mae_score, rmse_score, wmape_score