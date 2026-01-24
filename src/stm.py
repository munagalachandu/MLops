import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Loan Risk Predictor", page_icon="💰")

st.title("💰 Loan Risk Prediction App")

# -----------------------------
# Load Model Safely
# -----------------------------
MODEL_PATH = "loan_risk_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Sidebar Mode Selection
# -----------------------------
mode = st.sidebar.radio("Choose Prediction Mode", ["Single Prediction", "Batch Prediction (CSV)"])

# =========================================================
# 🔹 SINGLE PREDICTION
# =========================================================
if mode == "Single Prediction":
    st.header("🔍 Single Applicant Prediction")

    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Annual Income", min_value=1000, step=1000)
    employment = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
    residence = st.selectbox("Residence Type", ["Rented", "Owned", "Parental Home"])
    credit_score = st.slider("Credit Score", 300, 850, 650)
    loan_amount = st.number_input("Loan Amount", min_value=1000, step=1000)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    previous_default = st.radio("Previous Loan Default?", ["No", "Yes"])

    if st.button("Predict Risk"):
        input_data = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "EmploymentType": employment,
            "ResidenceType": residence,
            "CreditScore": credit_score,
            "LoanAmount": loan_amount,
            "LoanTerm": loan_term,
            "PreviousDefault": previous_default
        }])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data).max()

        st.subheader("📊 Prediction Result")

        if prediction == "High Risk":
            st.error(f"⚠️ {prediction}")
        else:
            st.success(f"✅ {prediction}")

        st.write(f"Confidence: **{proba*100:.2f}%**")

# =========================================================
# 📂 BATCH PREDICTION
# =========================================================
else:
    st.header("📂 Batch Prediction from CSV")

    st.write("Upload a CSV file with these columns:")
    st.code("""
Age,Income,EmploymentType,ResidenceType,CreditScore,LoanAmount,LoanTerm,PreviousDefault
""")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(batch_df.head())

            required_cols = [
                "Age", "Income", "EmploymentType", "ResidenceType",
                "CreditScore", "LoanAmount", "LoanTerm", "PreviousDefault"
            ]

            if not all(col in batch_df.columns for col in required_cols):
                st.error("❌ CSV is missing required columns.")
            else:
                predictions = model.predict(batch_df)
                probabilities = model.predict_proba(batch_df).max(axis=1)

                batch_df["PredictedRisk"] = predictions
                batch_df["Confidence"] = (probabilities * 100).round(2)

                st.subheader("✅ Prediction Results")
                st.dataframe(batch_df)

                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv,
                    file_name="loan_risk_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
