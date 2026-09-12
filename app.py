import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👨‍💼", layout="wide")

@st.cache_data
def make_dataset(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 61, n)
    income = rng.integers(18000, 150001, n)
    years = rng.integers(0, 16, n)
    overtime = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    satisfaction = rng.integers(1, 5, n)
    environment = rng.integers(1, 5, n)
    job_level = rng.integers(1, 6, n)
    distance = rng.integers(1, 31, n)
    jobs = rng.choice(["Sales", "IT", "HR", "Finance", "Operations"], n)
    travel = rng.choice(["Low", "Medium", "High"], n, p=[.45, .35, .20])

    score = (
        1.2*(overtime=="Yes") +
        0.9*(satisfaction<=2) +
        0.5*(environment<=2) +
        0.45*(distance>=20) +
        0.35*(years<=2) +
        0.35*(income<35000) +
        0.3*(travel=="High") -
        0.45*(job_level>=4) -
        0.25*(satisfaction>=4)
    )
    probability = 1 / (1 + np.exp(-(score - 1.9)))
    attrition = rng.random(n) < probability

    return pd.DataFrame({
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "OverTime": overtime,
        "JobSatisfaction": satisfaction,
        "EnvironmentSatisfaction": environment,
        "JobLevel": job_level,
        "DistanceFromHome": distance,
        "Department": jobs,
        "BusinessTravel": travel,
        "Attrition": np.where(attrition, "Yes", "No")
    })

@st.cache_resource
def train():
    df = make_dataset()
    X = df.drop(columns="Attrition")
    y = df["Attrition"]

    categorical = ["OverTime", "Department", "BusinessTravel"]
    numerical = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return df, model, accuracy, y_test, predictions

df, model, accuracy, y_test, predictions = train()

st.title("👨‍💼 Employee Attrition Prediction")
st.write("A medium-level Machine Learning project using Random Forest and a Streamlit dashboard.")

c1, c2, c3 = st.columns(3)
c1.metric("Dataset Rows", len(df))
c2.metric("Model Accuracy", f"{accuracy:.1%}")
c3.metric("Attrition Rate", f"{(df.Attrition == 'Yes').mean():.1%}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Employee Information")
    age = st.slider("Age", 20, 60, 30)
    income = st.slider("Monthly Income", 18000, 150000, 45000, 1000)
    years = st.slider("Years at Company", 0, 15, 3)
    overtime = st.selectbox("Overtime", ["No", "Yes"])
    satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    environment = st.slider("Environment Satisfaction", 1, 4, 3)
    level = st.slider("Job Level", 1, 5, 2)
    distance = st.slider("Distance From Home (km)", 1, 30, 8)
    department = st.selectbox("Department", ["Sales", "IT", "HR", "Finance", "Operations"])
    travel = st.selectbox("Business Travel", ["Low", "Medium", "High"])

with right:
    st.subheader("Prediction")
    input_df = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "OverTime": overtime,
        "JobSatisfaction": satisfaction,
        "EnvironmentSatisfaction": environment,
        "JobLevel": level,
        "DistanceFromHome": distance,
        "Department": department,
        "BusinessTravel": travel
    }])

    if st.button("🔮 Predict Attrition", use_container_width=True):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        yes_index = list(model.classes_).index("Yes")
        risk = prob[yes_index]

        if pred == "Yes":
            st.error(f"⚠️ High attrition risk — probability: {risk:.1%}")
        else:
            st.success(f"✅ Low attrition risk — probability: {risk:.1%}")

        st.progress(float(risk), text=f"Attrition probability: {risk:.1%}")

st.divider()
st.subheader("📊 Dataset Overview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Attrition by Overtime")
chart = pd.crosstab(df["OverTime"], df["Attrition"])
st.bar_chart(chart)

st.caption("Educational project. The dataset is synthetically generated for demonstration and is not suitable for real HR decisions.")
