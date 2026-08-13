from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# File Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed_data.csv")
MODEL_COMPARISON = os.path.join(BASE_DIR, "model_comparison.csv")


# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Dashboard
# -----------------------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# -----------------------------
# Prediction Page
# -----------------------------
@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# -----------------------------
# Explainability Page
# -----------------------------
@app.route("/explain")
def explain():
    return render_template("explain.html")


# -----------------------------
# About
# -----------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# =====================================================
# API : Dashboard Statistics
# =====================================================
@app.route("/api/stats")
def stats():

    if not os.path.exists(PROCESSED_DATA):
        return jsonify({"error": "processed_data.csv not found"}), 404

    df = pd.read_csv(PROCESSED_DATA)

    demand_column = None

    for col in df.columns:
        if col.lower() == "telangana":
            demand_column = col
            break

    if demand_column is None:
        return jsonify({"error": "Demand column not found"}), 404

    response = {
        "rows": len(df),
        "max_demand": round(df[demand_column].max(), 2),
        "min_demand": round(df[demand_column].min(), 2),
        "average_demand": round(df[demand_column].mean(), 2)
    }

    return jsonify(response)


# =====================================================
# API : Demand Trend Chart
# =====================================================
@app.route("/api/chart")
def chart():

    if not os.path.exists(PROCESSED_DATA):
        return jsonify([])

    df = pd.read_csv(PROCESSED_DATA)

    date_col = df.columns[0]
    demand_col = "Telangana"

    chart = []

    for _, row in df.iterrows():
        chart.append({
            "date": str(row[date_col]),
            "demand": float(row[demand_col])
        })

    return jsonify(chart)


# =====================================================
# API : Model Comparison
# =====================================================
@app.route("/api/models")
def models():

    if not os.path.exists(MODEL_COMPARISON):
        return jsonify([])

    df = pd.read_csv(MODEL_COMPARISON)

    return jsonify(df.to_dict(orient="records"))


# =====================================================
# API : Feature Importance
# =====================================================
@app.route("/api/features")
def features():

    data = [

        {
            "feature": "Demand Lag 1",
            "importance": 52,
            "description": "Previous day's demand."
        },

        {
            "feature": "Demand Lag 7",
            "importance": 23,
            "description": "Same weekday previous week."
        },

        {
            "feature": "Month",
            "importance": 9,
            "description": "Seasonality."
        },

        {
            "feature": "Day of Week",
            "importance": 7,
            "description": "Weekly demand cycle."
        },

        {
            "feature": "IPL Day",
            "importance": 5,
            "description": "IPL match impact."
        },

        {
            "feature": "Festival",
            "importance": 3,
            "description": "Festival demand variation."
        },

        {
            "feature": "Weekend",
            "importance": 1,
            "description": "Weekend effect."
        }

    ]

    return jsonify(data)


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)