import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from backend.predict import predict_scores
from backend.shap_explainer import generate_shap_plot

st.set_page_config(page_title="EduPredict Pro", layout="wide")

st.title("🎓 EduPredict Pro")
st.subheader("Advanced Student Performance Prediction System")

uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(input_df.head())

    reg_pred, clf_pred = predict_scores(input_df)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Final Score", round(reg_pred, 2))

    with col2:
        st.metric("Predicted Result", clf_pred)

    st.write("### SHAP Feature Importance")

    shap_values = generate_shap_plot(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
