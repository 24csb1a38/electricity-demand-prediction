"""
main.py
Runs the full pipeline end to end:
1. Load + process data
2. Evaluate naive baseline
3. Train + tune XGBoost with Optuna
4. Train Prophet
5. Explain the best model with SHAP
6. Print a final comparison table

Run this with:  python main.py
(make sure your virtual environment is activated first)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "notebooks", "src"))

import pandas as pd

from data_prep import build_features, DEMAND_COLUMN, DATE_COLUMN
from baseline import evaluate_baseline
from xgboost_model import time_based_split, tune_xgboost_with_optuna
from prophet_model import train_prophet
from explain import explain_model


def main():
    print("Step 1: Building features...")
    df = build_features()

    print("\nStep 2: Evaluating naive baseline...")
    baseline_results = evaluate_baseline(df, demand_column=DEMAND_COLUMN)

    print("\nStep 3: Training + tuning XGBoost with Optuna...")
    X_train, X_test, y_train, y_test = time_based_split(df, demand_column=DEMAND_COLUMN)
    xgb_results = tune_xgboost_with_optuna(X_train, y_train, X_test, y_test, n_trials=30)

    print("\nStep 4: Training Prophet...")
    prophet_results = train_prophet(df, date_column=DATE_COLUMN, demand_column=DEMAND_COLUMN)

    print("\nStep 5: Explaining XGBoost model with SHAP...")
    explain_model(xgb_results["trained_model"], X_test)

    print("\n=== FINAL COMPARISON TABLE ===")
    comparison = pd.DataFrame([
        baseline_results,
        {"model": xgb_results["model"], "MAE": xgb_results["MAE"], "RMSE": xgb_results["RMSE"]},
        {"model": prophet_results["model"], "MAE": prophet_results["MAE"], "RMSE": prophet_results["RMSE"]},
    ])
    print(comparison.to_string(index=False))
    comparison.to_csv("model_comparison.csv", index=False)
    print("\nSaved comparison table to model_comparison.csv")


if __name__ == "__main__":
    main()
