import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from backend.predict import predict_scores

st.set_page_config(page_title="EduPredict Pro", layout="wide")

st.title("🎓 Student Result Prediction System")
st.write("Predict Final Exam Score & Pass/Fail Status")

# ===============================
# INPUT SECTION
# ===============================

def user_input():
    data = {
        "attendance_percentage": st.slider("Attendance (%)", 40, 100, 75),
        "internal_exam_1": st.slider("Internal Exam 1", 30, 100, 70),
        "internal_exam_2": st.slider("Internal Exam 2", 30, 100, 70),
        "assignment_score_avg": st.slider("Assignment Average", 40, 100, 75),
        "quiz_avg": st.slider("Quiz Average", 30, 100, 70),
        "lab_performance": st.slider("Lab Performance", 40, 100, 75),
        "previous_sem_gpa": st.slider("Previous GPA", 4.0, 10.0, 7.0),
        "backlog_count": st.slider("Backlogs", 0, 5, 0),
        "project_score": st.slider("Project Score", 40, 100, 75),
        "midterm_score": st.slider("Midterm Score", 40, 100, 70),
        "study_hours_per_day": st.slider("Study Hours/Day", 0.5, 10.0, 3.0),
        "library_visits_per_month": st.slider("Library Visits/Month", 0, 20, 5),
        "online_course_count": st.slider("Online Courses", 0, 10, 2),
        "doubt_sessions_attended": st.slider("Doubt Sessions", 0, 10, 3),
        "age": st.slider("Age", 18, 25, 20),
        "family_income": st.number_input("Family Income", value=500000),
        "distance_from_college_km": st.slider("Distance (km)", 1, 40, 10),
        "sleep_hours": st.slider("Sleep Hours", 4.0, 9.0, 6.5),
        "stress_level": st.slider("Stress Level", 1.0, 10.0, 5.0),
        "social_media_hours": st.slider("Social Media Hours", 0.5, 8.0, 3.0),
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "parental_education_level": st.selectbox(
            "Parental Education", ["School", "Graduate", "Postgraduate"]
        ),
        "urban_or_rural": st.selectbox("Location", ["Urban", "Rural"]),
        "internet_access_home": st.selectbox("Internet Access", ["Yes", "No"]),
        "participation_club": st.selectbox("Club Participation", ["Yes", "No"]),
        "participation_sports": st.selectbox("Sports Participation", ["Yes", "No"]),
        "internship_completed": st.selectbox("Internship Completed", ["Yes", "No"]),
        "health_issues": st.selectbox("Health Issues", ["Yes", "No"]),
        "part_time_job": st.selectbox("Part Time Job", ["Yes", "No"]),
    }
    return data


input_data = user_input()

# ===============================
# PREDICTION
# ===============================

if st.button("Predict Result"):
    score, status = predict_scores(input_data)

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Final Score: {round(score,2)}")
    st.info(f"Pass/Fail Status: {status}")
