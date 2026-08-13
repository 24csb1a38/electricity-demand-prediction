"""
prophet_model.py
Trains a Facebook Prophet model on the demand time series,
using festival + IPL dates as built-in "holiday" regressors.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prep import FESTIVAL_DATES, IPL_DATE_RANGES


def build_holidays_df():
    """
    Prophet expects a dataframe with columns: holiday, ds (date)
    We combine festival dates and IPL dates into this format.
    """
    holiday_rows = []

    for date in FESTIVAL_DATES:
        holiday_rows.append({"holiday": "festival", "ds": date})

    for start, end in IPL_DATE_RANGES:
        for date in pd.date_range(start, end):
            holiday_rows.append({"holiday": "ipl", "ds": date.strftime("%Y-%m-%d")})

    holidays_df = pd.DataFrame(holiday_rows)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    return holidays_df


def train_prophet(df, date_column="Date", demand_column="Demand", test_size=0.2):
    """
    Prophet requires a dataframe with exactly two columns: 'ds' (date) and 'y' (target).
    """
    prophet_df = df[[date_column, demand_column]].rename(
        columns={date_column: "ds", demand_column: "y"}
    )

    split_idx = int(len(prophet_df) * (1 - test_size))
    train_df = prophet_df.iloc[:split_idx]
    test_df = prophet_df.iloc[split_idx:]

    holidays_df = build_holidays_df()

    model = Prophet(holidays=holidays_df, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(train_df)

    future = test_df[["ds"]]
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"Prophet -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return {"model": "Prophet", "MAE": mae, "RMSE": rmse, "trained_model": model, "forecast": forecast}
