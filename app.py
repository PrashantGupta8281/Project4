import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load and Train Model
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("HR_comma_sep.csv")

    X = df[['satisfaction_level',
            'average_montly_hours',
            'promotion_last_5years',
            'salary']]

    X = pd.get_dummies(X, columns=['salary'])

    y = df['left']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, X.columns, accuracy


model, columns, accuracy = train_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Employee Retention Prediction")

st.write(f"Model Accuracy: **{accuracy:.2%}**")

satisfaction = st.slider(
    "Satisfaction Level",
    0.0, 1.0, 0.5
)

hours = st.number_input(
    "Average Monthly Hours",
    min_value=80,
    max_value=350,
    value=200
)

promotion = st.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.selectbox(
    "Salary Level",
    ["low", "medium", "high"]
)

if st.button("Predict"):

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=columns
    )

    input_data["satisfaction_level"] = satisfaction
    input_data["average_montly_hours"] = hours
    input_data["promotion_last_5years"] = promotion

    input_data[f"salary_{salary}"] = 1

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error("The employee is likely to leave the company.")
    else:
        st.success("The employee is likely to stay with the company.")

    st.write(f"Probability of Staying: {probability[0]:.2%}")
    st.write(f"Probability of Leaving: {probability[1]:.2%}")
