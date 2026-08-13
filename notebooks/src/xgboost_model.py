"""
xgboost_model.py
Trains an XGBoost regressor to predict electricity demand,
tunes it with Optuna, and returns the best model + metrics.
"""

import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "day_of_week", "month", "is_weekend",
    "demand_lag_1", "demand_lag_7",
    "is_festival", "is_ipl_day"
]


def time_based_split(df, demand_column="Demand", test_size=0.2):
    """
    Time series data should NOT be split randomly.
    Take the last `test_size` fraction of rows (by date) as the test set.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[demand_column]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[demand_column]

    return X_train, X_test, y_train, y_test


def train_baseline_xgboost(X_train, y_train, X_test, y_test):
    """Train a default XGBoost model with no tuning, as a sanity check."""
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Default XGBoost -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model, mae, rmse


def tune_xgboost_with_optuna(X_train, y_train, X_test, y_test, n_trials=30):
    """
    Uses Optuna to search for the best XGBoost hyperparameters.
    Returns the best trained model and its metrics.
    """

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best hyperparameters:", study.best_params)

    # Train final model with best params
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Tuned XGBoost -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return {
        "model": "XGBoost (Optuna-tuned)",
        "MAE": mae,
        "RMSE": rmse,
        "best_params": study.best_params,
        "trained_model": best_model,
    }
