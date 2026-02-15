import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression

def train_models():

    data = pd.read_csv("data/student_performance_advanced.csv")

    y_reg = data["final_exam_score"]
    y_clf = data["pass_fail"]
    X = data.drop(columns=["final_exam_score", "pass_fail"])

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    reg_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression, k=20)),
        ("model", RandomForestRegressor(random_state=42))
    ])

    clf_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg_pipeline.fit(X_train, y_train_reg)
    clf_pipeline.fit(X_train, y_clf.iloc[y_train_reg.index])

    return reg_pipeline, clf_pipeline
