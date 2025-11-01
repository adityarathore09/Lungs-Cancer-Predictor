import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# -----------------------------#
#      Page Configuration
# -----------------------------#
st.set_page_config(
    page_title="Lung Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------#
#  Custom Hospital Style CSS
# -----------------------------#
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0f7fa, #b2ebf2);
        color: #004d40;
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 42px;
        color: #006064;
        font-weight: 700;
        margin-top: 30px;
    }
    .sub {
        text-align: center;
        color: #00796b;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #00838f;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #006064;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------#
#      Load Data
# -----------------------------#
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")  # ✅ make sure dataset.csv is in same folder
    df.columns = df.columns.str.strip()
    df["GENDER"] = df["GENDER"].map({'M': 0, 'F': 1})
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({'YES': 1, 'NO': 0})
    return df

df = load_data()

# -----------------------------#
#     Train the Model
# -----------------------------#
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------#
#     Navigation System
# -----------------------------#
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user" not in st.session_state:
    st.session_state.user = ""

# -----------------------------#
#         LOGIN PAGE
# ---------------------- LOGIN PAGE ---------------------- #
if st.session_state.page == "login":
    st.markdown("<h1 class='main-title'>🩺 Lung Cancer Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub'>Your trusted AI health assistant</p>", unsafe_allow_html=True)

    # ---- Custom CSS for smaller input boxes ----
    st.markdown("""
        <style>
        div[data-baseweb="input"] > div:first-child {
            width: 300px !important;   /* makes input smaller */
        }
        </style>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        if st.button("Login"):
            if username and password:
                st.session_state.user = username
                st.session_state.page = "predict"
                st.rerun()
            else:
                st.error("Please enter both username and password.")
        if st.button("Sign Up"):
            st.success("Account created successfully! Please log in.")


#       PREDICTION PAGE
# -----------------------------#
elif st.session_state.page == "predict":
   
    st.markdown(f"<h1 class='main-title'>Welcome, {st.session_state.user}</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub'>Please fill in your medical details below to check your risk.</p>", unsafe_allow_html=True)
    st.sidebar.header("Patient Details")

    GENDER = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    AGE = st.sidebar.slider("Age", 20, 100, 50)
    SMOKING = st.sidebar.selectbox("Smoking (1=Yes, 2=No)", [1, 2])
    YELLOW_FINGERS = st.sidebar.selectbox("Yellow Fingers (1=Yes, 2=No)", [1, 2])
    ANXIETY = st.sidebar.selectbox("Anxiety (1=Yes, 2=No)", [1, 2])
    PEER_PRESSURE = st.sidebar.selectbox("Peer Pressure (1=Yes, 2=No)", [1, 2])
    CHRONIC_DISEASE = st.sidebar.selectbox("Chronic Disease (1=Yes, 2=No)", [1, 2])
    FATIGUE = st.sidebar.selectbox("Fatigue (1=Yes, 2=No)", [1, 2])
    ALLERGY = st.sidebar.selectbox("Allergy (1=Yes, 2=No)", [1, 2])
    WHEEZING = st.sidebar.selectbox("Wheezing (1=Yes, 2=No)", [1, 2])
    ALCOHOL_CONSUMING = st.sidebar.selectbox("Alcohol Consuming (1=Yes, 2=No)", [1, 2])
    COUGHING = st.sidebar.selectbox("Coughing (1=Yes, 2=No)", [1, 2])
    SHORTNESS_OF_BREATH = st.sidebar.selectbox("Shortness of Breath (1=Yes, 2=No)", [1, 2])
    SWALLOWING_DIFFICULTY = st.sidebar.selectbox("Swallowing Difficulty (1=Yes, 2=No)", [1, 2])
    CHEST_PAIN = st.sidebar.selectbox("Chest Pain (1=Yes, 2=No)", [1, 2])

    GENDER = 0 if GENDER == 'Male' else 1
    input_data = pd.DataFrame({
        'GENDER': [GENDER],
        'AGE': [AGE],
        'SMOKING': [SMOKING],
        'YELLOW_FINGERS': [YELLOW_FINGERS],
        'ANXIETY': [ANXIETY],
        'PEER_PRESSURE': [PEER_PRESSURE],
        'CHRONIC_DISEASE': [CHRONIC_DISEASE],
        'FATIGUE': [FATIGUE],
        'ALLERGY': [ALLERGY],
        'WHEEZING': [WHEEZING],
        'ALCOHOL_CONSUMING': [ALCOHOL_CONSUMING],
        'COUGHING': [COUGHING],
        'SHORTNESS_OF_BREATH': [SHORTNESS_OF_BREATH],
        'SWALLOWING_DIFFICULTY': [SWALLOWING_DIFFICULTY],
        'CHEST_PAIN': [CHEST_PAIN]
    })

    st.write("### Patient Input Summary")
    st.dataframe(input_data)

    if st.button("🔍 Predict Lung Cancer Risk"):
        pred = model.predict(input_data)[0]
        result = "⚠️ High Risk of Lung Cancer Detected" if pred == 1 else "✅ No Lung Cancer Risk Detected"

        st.session_state.prediction = result
        st.session_state.page = "thankyou"
        st.rerun()

# -----------------------------#
#        THANK YOU PAGE
# -----------------------------#
elif st.session_state.page == "thankyou":
    st.markdown(f"<h1 class='main-title'>Thank You, {st.session_state.user}!</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; color:#004d40;'>{st.session_state.prediction}</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("🔁 Check Again"):
            st.session_state.page = "predict"
            st.rerun()
        if st.button("🚪 Logout"):
            st.session_state.page = "login"
            st.rerun()
