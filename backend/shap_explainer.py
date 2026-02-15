import shap
import joblib
import pandas as pd

reg_model = joblib.load("models/regression_model.pkl")

def generate_shap_plot(input_df):
    explainer = shap.Explainer(reg_model)
    shap_values = explainer(input_df)

    return shap_values
