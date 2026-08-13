"""
explain.py
Uses SHAP to explain which features drive the XGBoost model's predictions.
Generates a summary plot (saved as an image) showing feature importance.
"""

import shap
import matplotlib.pyplot as plt


def explain_model(trained_model, X_test, output_path="shap_summary_plot.png"):
    """
    Generates and saves a SHAP summary plot for the given trained model.
    """
    explainer = shap.Explainer(trained_model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"SHAP summary plot saved to {output_path}")

    # Also print average absolute SHAP value per feature (a simple ranked table)
    import numpy as np
    import pandas as pd

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nFeature importance (by mean absolute SHAP value):")
    print(importance_df.to_string(index=False))

    return importance_df
