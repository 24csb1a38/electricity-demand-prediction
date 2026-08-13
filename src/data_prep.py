"""
data_prep.py
Loads raw power consumption data, cleans it, and engineers features:
- date-based features (day of week, month, weekend flag)
- lag features (demand 1 day ago, 7 days ago)
- event features (festival flag, IPL match day flag)

IMPORTANT: Adjust COLUMN NAMES below to match your actual Kaggle/POSOCO CSV.
Different versions of the dataset name columns differently
(e.g. "Date", "date", "Demand", "Usage", "States" vs a single state column).
"""

import pandas as pd
import numpy as np

# ---- CONFIG: change these to match your CSV ----
RAW_DATA_PATH = "data/raw_power_consumption.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"

DATE_COLUMN = "Unnamed: 0"
DEMAND_COLUMN = "Telangana"
STATE_COLUMN = None          # change if your date column has a different name        # set to None if your dataset has no state column
STATE_FILTER = "Telangana"    # which state to filter to, if STATE_COLUMN is set

# ---- Example festival dates (EDIT to match the exact years your dataset covers) ----
# These are placeholders - verify actual dates for your dataset's year range
# before using them, since festival dates change every year.
FESTIVAL_DATES = [
    "2019-10-27",  # Diwali 2019 (example - verify)
    "2019-10-08",  # Dussehra 2019 (example - verify)
    "2020-11-14",  # Diwali 2020 (example - verify)
    "2020-10-25",  # Dussehra 2020 (example - verify)
]

# Example IPL date ranges (EDIT to match your data's years)
# IPL typically runs March-May; add exact match dates if you want day-level precision
IPL_DATE_RANGES = [
    ("2019-03-23", "2019-05-12"),
    ("2020-09-19", "2020-11-10"),  # 2020 IPL was delayed to UAE, Sep-Nov
]


def load_raw_data():
    """Load the raw CSV and do basic cleaning."""
    df = pd.read_csv(RAW_DATA_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], dayfirst=True)

    if STATE_COLUMN and STATE_COLUMN in df.columns:
        df = df[df[STATE_COLUMN] == STATE_FILTER].copy()

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Drop rows with missing demand values
    df = df.dropna(subset=[DEMAND_COLUMN])

    return df


def add_calendar_features(df):
    """Add day-of-week, month, weekend flag."""
    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek  # 0=Monday
    df["month"] = df[DATE_COLUMN].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df):
    """Add lag features: demand 1 day ago, 7 days ago."""
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df["demand_lag_1"] = df[DEMAND_COLUMN].shift(1)
    df["demand_lag_7"] = df[DEMAND_COLUMN].shift(7)
    # Drop rows where lag features are NaN (first 7 rows)
    df = df.dropna(subset=["demand_lag_1", "demand_lag_7"]).reset_index(drop=True)
    return df


def add_event_features(df):
    """Add is_festival and is_ipl_day flags based on the date lists above."""
    festival_dates = pd.to_datetime(FESTIVAL_DATES)
    df["is_festival"] = df[DATE_COLUMN].isin(festival_dates).astype(int)

    def is_ipl_day(date):
        for start, end in IPL_DATE_RANGES:
            if pd.to_datetime(start) <= date <= pd.to_datetime(end):
                return 1
        return 0

    df["is_ipl_day"] = df[DATE_COLUMN].apply(is_ipl_day)
    return df


def build_features():
    """Run the full pipeline: load, clean, engineer features, save."""
    df = load_raw_data()
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_event_features(df)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(df.head())
    return df


if __name__ == "__main__":
    build_features()
