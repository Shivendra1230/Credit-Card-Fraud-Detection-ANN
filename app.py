import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
}

.metric-box{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 0px 10px rgba(255,255,255,0.1);
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-card-back-side.png",
    width=90
)

st.sidebar.title("Credit Card Fraud Detection")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "🎲 Demo Prediction",

        "📂 CSV Upload",

        "ℹ Model Information"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Application Loaded Successfully")
st.sidebar.markdown("---")

st.sidebar.info("""
### Model Details

✅ ANN

✅ TensorFlow

✅ Hyperparameter Tuned

✅ Threshold = 0.99

""")
# -------------------------
# HEADER
# -------------------------
# -------------------------
# LOAD MODEL & SCALER
# -------------------------

@st.cache_resource
def load_assets():

    model = load_model("models/credit_card_fraud_model.keras")

    scaler = joblib.load("models/scaler.pkl")

    return model, scaler

try:

    with st.spinner("Loading ANN Model..."):

        model, scaler = load_assets()

except Exception as e:

    st.error(f"Error loading model : {e}")

    st.stop()
st.title("💳 Credit Card Fraud Detection")

st.caption(
    "Artificial Neural Network based Fraud Detection System"
)

st.divider()

# -------------------------
# METRIC CARDS
# -------------------------

col1,col2,col3 = st.columns(3)

with col1:

    st.metric(

        "ROC-AUC",

        "96.6%",

        "Excellent"

    )

with col2:

    st.metric(

        "Precision",

        "87%",

        "High"

    )

with col3:

    st.metric(

        "Recall",

        "76%",

        "Good"

    )

st.divider()

# -------------------------
# HOME PAGE
# -------------------------

if page=="🏠 Home":

    st.header("Welcome 👋")

    st.write("""

This project predicts fraudulent credit card transactions using an Artificial Neural Network.

### Features

- Artificial Neural Network
- Hyperparameter Tuning
- Threshold Tuning
- CSV Prediction
- Random Demo Prediction
- Download Prediction CSV
- Interactive Dashboard

""")

    c1,c2,c3 = st.columns(3)

    c1.metric("Model","ANN")

    c2.metric("Features","30")

    c3.metric("Classes","Fraud/Genuine")
    # =====================================================
# DEMO PREDICTION
# =====================================================

elif page == "🎲 Demo Prediction":
    

    st.header("🎲 Demo Prediction")

    st.write("Click the button below to test the model on a random transaction from the dataset.")

    if st.button("🚀 Generate Random Transaction"):

        # Read Dataset
        df = pd.read_csv("demo_transactions.csv")

        # Random Sample
        sample = df.sample(1)

        # Remove Target Column
        X = sample.drop(columns=["Class"], errors="ignore")

        # Scale Features
        X_scaled = scaler.transform(X)

        # Prediction Probability
        probability = float(model.predict(X_scaled, verbose=0)[0][0])

        THRESHOLD = 0.99

        st.divider()

        # -----------------------------
        # Probability Gauge
        # -----------------------------

        col1, col2 = st.columns([2,1])

        with col1:

            st.subheader("Prediction Result")

            st.progress(probability)

            st.write(f"### Fraud Probability : {probability:.2%}")

            confidence = max(probability, 1-probability)

            st.info(f"Confidence : {confidence:.2%}")

            if probability > THRESHOLD:

                st.error("🚨 Fraud Transaction")

            else:

                st.success("✅ Genuine Transaction")

        with col2:

            fig = go.Figure(go.Indicator(

                mode="gauge+number",

                value=probability*100,

                title={"text":"Fraud Probability"},

                gauge={

                    "axis":{"range":[0,100]},

                    "bar":{"color":"red"},

                    "steps":[

                        {"range":[0,40],"color":"green"},

                        {"range":[40,70],"color":"yellow"},

                        {"range":[70,100],"color":"red"}

                    ]

                }

            ))

            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Show Transaction
        # -----------------------------

        with st.expander("📋 Show Transaction Details"):

            st.dataframe(sample.T)

        # -----------------------------
        # Actual Label
        # -----------------------------

        actual = int(sample["Class"].values[0])

        st.write("### Actual Label")

        if actual == 1:

            st.error("Actual : Fraud")

        else:

            st.success("Actual : Genuine")
    # =====================================================
# CSV UPLOAD
# =====================================================

elif page == "📂 CSV Upload":

    st.header("📂 Upload Transaction CSV")

    st.write(
        "Upload a CSV file containing credit card transactions to predict fraud."
    )

    # --------------------------------------------------
    # Download Sample CSV
    # --------------------------------------------------

    sample_df = pd.read_csv("data/demo_transactions.csv").head(10)

    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            required_columns = [
                "Time",
                "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                "V11","V12","V13","V14","V15","V16","V17","V18","V19",
                "V20","V21","V22","V23","V24","V25","V26","V27","V28",
                "Amount"
            ]

            missing = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if len(missing) > 0:

                st.error("❌ Invalid CSV Format")

                st.warning(
                    "Missing Columns:\n\n"
                    + ", ".join(missing)
                )

                st.info(
                    "Download the sample CSV and use the same format."
                )

                st.stop()

            st.success("✅ Valid CSV Uploaded")

            st.subheader("Dataset Preview")

            st.dataframe(df.head())

            c1, c2 = st.columns(2)

            c1.metric(
                "Rows",
                df.shape[0]
            )

            c2.metric(
                "Columns",
                df.shape[1]
            )

            st.divider()

            if st.button("🚀 Predict Fraud"):

                X = df[required_columns]

                X_scaled = scaler.transform(X)

                probability = model.predict(
                    X_scaled,
                    verbose=0
                ).flatten()

                threshold = 0.99

                prediction = np.where(
                    probability > threshold,
                    "Fraud",
                    "Genuine"
                )

                df["Fraud Probability"] = probability

                df["Prediction"] = prediction

                fraud = (prediction == "Fraud").sum()

                genuine = (prediction == "Genuine").sum()

                st.subheader("📊 Prediction Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Transactions",
                    len(df)
                )

                col2.metric(
                    "Fraud",
                    fraud
                )

                col3.metric(
                    "Genuine",
                    genuine
                )

                st.progress(
                    fraud / len(df)
                )

                st.info(
                    f"Fraud Rate : {(fraud/len(df))*100:.2f}%"
                )

                st.divider()

                st.subheader("Prediction Result")

                st.dataframe(df)

                st.download_button(
                    label="📥 Download Prediction CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

                st.success(
                    "Prediction Completed Successfully ✅"
                )

        except Exception as e:

            st.error(
                f"Error while processing file:\n\n{e}"
            )


elif page=="ℹ Model Information":

    st.header("ℹ Model Information")

    st.markdown("""
### Model
- Artificial Neural Network (ANN)

### Optimizer
- Adam

### Loss
- Binary Crossentropy

### Hidden Layers
- 96 → 64 → 1

### Activation
- ReLU + Sigmoid

### Threshold
- 0.99

### ROC-AUC
- 96.6%
""")

st.divider()

st.markdown("""
<div style='text-align:center'>
<h3>💳 Credit Card Fraud Detection</h3>
<p><b>Developed by Shivendra Pratap Singh</b></p>
<p>TensorFlow • Keras • Streamlit</p>
</div>
""", unsafe_allow_html=True)
