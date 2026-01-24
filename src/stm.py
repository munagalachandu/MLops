import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("loan_risk_model.pkl")

st.set_page_config(page_title="Loan Risk Predictor", page_icon="💰")

st.title("💰 Loan Risk Prediction App")
st.write("Enter applicant details to predict loan risk category.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.slider("Age", 18, 70, 30)
income = st.number_input("Annual Income", min_value=1000, step=1000)
employment = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
residence = st.selectbox("Residence Type", ["Rented", "Owned", "Parental Home"])
credit_score = st.slider("Credit Score", 300, 850, 650)
loan_amount = st.number_input("Loan Amount", min_value=1000, step=1000)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
previous_default = st.radio("Previous Loan Default?", ["No", "Yes"])

# -----------------------------
# Prediction Button
# -----------------------------
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
