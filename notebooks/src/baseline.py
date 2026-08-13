"""
baseline.py
The naive baseline: predict demand = same value as 7 days ago.
Every real model must beat this to be worth anything.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_baseline(df, demand_column="Demand"):
    """
    Uses the demand_lag_7 column (already created in data_prep.py)
    as the prediction, compares it to actual demand.
    """
    y_true = df[demand_column]
    y_pred = df["demand_lag_7"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("=== Naive Baseline (same day last week) ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return {"model": "Naive Baseline", "MAE": mae, "RMSE": rmse}
