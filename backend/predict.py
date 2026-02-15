import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

reg_model_path = os.path.join(BASE_DIR, "models", "regression_model(2).pkl")
clf_model_path = os.path.join(BASE_DIR, "models", "classification_model(2).pkl")

reg_model = joblib.load(reg_model_path)
clf_model = joblib.load(clf_model_path)

def predict_scores(input_dict):
    input_df = pd.DataFrame([input_dict])
    
    reg_prediction = reg_model.predict(input_df)[0]
    clf_prediction = clf_model.predict(input_df)[0]
    
    return reg_prediction, clf_prediction
