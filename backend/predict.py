import joblib
import pandas as pd

reg_model = joblib.load("models/regression_model.pkl")
clf_model = joblib.load("models/classification_model.pkl")

def predict_scores(input_df):
    reg_prediction = reg_model.predict(input_df)[0]
    clf_prediction = clf_model.predict(input_df)[0]

    return reg_prediction, clf_prediction
