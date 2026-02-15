import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from model import train_models

st.set_page_config(page_title="EduPredict Pro", layout="wide")

st.title("🎓 EduPredict Pro")

@st.cache_resource
def load_models():
    return train_models()

reg_model, clf_model = load_models()

uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(input_df.head())

    reg_prediction = reg_model.predict(input_df)[0]
    clf_prediction = clf_model.predict(input_df)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Final Score", round(reg_prediction, 2))

    with col2:
        st.metric("Predicted Result", clf_prediction)

    st.write("### SHAP Explainability")

    explainer = shap.Explainer(reg_model.named_steps["model"])
    shap_values = explainer(reg_model.named_steps["preprocessing"].transform(input_df))

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
