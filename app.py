import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="💼",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#111827);
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#2563eb,#1d4ed8);
}

/* Buttons */
.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#06b6d4,#2563eb);
    color:white;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    padding:12px;
    transition:0.3s;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#3b82f6,#8b5cf6);
    transform:scale(1.03);
}

/* Prediction Cards */
.result-card{
    background:#1e293b;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 8px 30px rgba(0,0,0,0.3);
    margin-top:20px;
}

/* Header */
.main-title{
    text-align:center;
    color:#38bdf8;
    font-size:48px;
    font-weight:700;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:30px;
}

/* Footer */
.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TRAIN MODEL
# ==========================================================
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
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, X.columns, accuracy


model, columns, accuracy = train_model()

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
"""
<div class='main-title'>
💼 Employee Retention Prediction
</div>

<div class='sub-title'>
Predict whether an employee is likely to stay or leave using Machine Learning
</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# MODEL ACCURACY
# ==========================================================
c1, c2, c3 = st.columns(3)

with c2:
    st.metric("🎯 Model Accuracy", f"{accuracy*100:.2f}%")

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("📝 Employee Details")

satisfaction = st.sidebar.slider(
    "😊 Satisfaction Level",
    0.0,
    1.0,
    0.50
)

hours = st.sidebar.slider(
    "⏰ Average Monthly Hours",
    80,
    350,
    200
)

promotion = st.sidebar.selectbox(
    "🚀 Promotion in Last 5 Years",
    [0,1]
)

salary = st.sidebar.selectbox(
    "💰 Salary Level",
    ["low","medium","high"]
)

predict = st.sidebar.button("🔍 Predict")

st.sidebar.title("My Portfolio")

st.sidebar.markdown("""
### 👋 About Me

**Name:** Prashant Gupta

🔗 **LinkedIn:** [Click Here](https://www.linkedin.com/in/prashant-gupta-012320389?utm_source=share_via&utm_content=profile&utm_medium=member_android)

💻 **GitHub:** [Click Here](https://github.com/PrashantGupta8281/AI-ML_Summer_Internship)
""")

# ==========================================================
# MAIN CONTENT
# ==========================================================
left_col, right_col = st.columns([1,1])

with left_col:

    st.subheader("📋 Employee Information")

    st.info(f"""
**Satisfaction Level:** {satisfaction}

**Average Monthly Hours:** {hours}

**Promotion:** {promotion}

**Salary Level:** {salary.title()}
""")

with right_col:

    st.subheader("📈 Prediction")

    if predict:

        input_df = pd.DataFrame(
            0,
            index=[0],
            columns=columns
        )

        input_df["satisfaction_level"] = satisfaction
        input_df["average_montly_hours"] = hours
        input_df["promotion_last_5years"] = promotion
        input_df[f"salary_{salary}"] = 1

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        if prediction == 1:

            st.markdown("""
            <div class='result-card'>
            <h2 style='color:#ef4444'>
            ❌ Employee is Likely to Leave
            </h2>

            <h4 style='color:white'>
            The employee has a higher probability of leaving the company.
            Consider improving engagement, work-life balance, and career opportunities.
            </h4>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='result-card'>
            <h2 style='color:#22c55e'>
            ✅ Employee is Likely to Stay
            </h2>

            <h4 style='color:white'>
            The employee appears satisfied and is likely to remain with the organization.
            </h4>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.subheader("📊 Prediction Confidence")

        p1, p2 = st.columns(2)

        with p1:
            st.metric(
                "😊 Stay Probability",
                f"{probability[0]*100:.2f}%"
            )

        with p2:
            st.metric(
                "🚪 Leave Probability",
                f"{probability[1]*100:.2f}%"
            )

        st.write("### Employee Leaving Risk")

        st.progress(float(probability[1]))

    else:

        st.info("👈 Enter employee details from the sidebar and click **Predict**.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown(
"""
<div class='footer'>
<hr>
<h4>💼 Employee Retention Prediction System</h4>
<p>Built using <b>Streamlit</b> • <b>Scikit-Learn</b> • <b>Pandas</b></p>
</div>
""",
unsafe_allow_html=True
)
