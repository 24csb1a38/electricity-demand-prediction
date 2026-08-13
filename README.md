# Electricity Demand Spike Prediction Using Grid + Event Data

Time-series ML system predicting electricity demand spikes in Telangana by
correlating real state-wise power consumption data with festival and major
sporting event (IPL) calendars.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Prophet, Optuna, SHAP, Matplotlib

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle (search "India power consumption" or
   "POSOCO state-wise power consumption") and place the CSV at:
   ```
   data/raw_power_consumption.csv
   ```

4. **Important:** Open `src/data_prep.py` and adjust `DATE_COLUMN`,
   `DEMAND_COLUMN`, and `STATE_COLUMN` to match your actual CSV's column
   names. Also verify/update the festival and IPL dates to match the
   years your dataset actually covers.

## Running the pipeline

```
python main.py
```

This will:
1. Clean the data and engineer features (lag features, calendar features, event flags)
2. Evaluate a naive baseline (same-day-last-week prediction)
3. Train and tune an XGBoost model using Optuna
4. Train a Prophet model with festival/IPL dates as holidays
5. Generate a SHAP summary plot explaining feature importance
6. Print and save a final comparison table (`model_comparison.csv`)

## Project Structure

```
electricity-demand-prediction/
├── data/                  # raw and processed data (raw CSV not tracked in git)
├── src/
│   ├── data_prep.py       # data cleaning + feature engineering
│   ├── baseline.py        # naive baseline model
│   ├── xgboost_model.py   # XGBoost training + Optuna tuning
│   ├── prophet_model.py   # Prophet training
│   └── explain.py         # SHAP explainability
├── main.py                # runs the full pipeline
├── requirements.txt
└── README.md
```
