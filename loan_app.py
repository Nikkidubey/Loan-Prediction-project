import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the improved model and scaler
try:
    model = joblib.load("improved_loan_model.pkl")
    scaler = joblib.load("improved_loan_scaler.pkl")
    feature_cols = joblib.load("improved_loan_features.pkl")
    st.success("✅ Improved model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Streamlit UI
st.title("🏦 Loan Default Prediction Dashboard")
st.write("Enter applicant details to predict loan default risk.")

# Display model info
st.info(f"Model uses {len(feature_cols)} features: {', '.join(feature_cols)}")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Financial Information")
    income = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0, value=50000.0, help="Applicant's annual income")
    loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, step=1000.0, value=30000.0, help="Amount requested for loan")
    debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01, help="Monthly debt payments / Monthly income")
    loan_to_income = st.slider("Loan-to-Income Ratio", 0.0, 1.0, 0.2, 0.01, help="Loan amount / Annual income")

with col2:
    st.subheader("👤 Personal Information")
    credit_score = st.slider("Credit Score (0-1)", 0.0, 1.0, 0.5, 0.01, help="Normalized credit score (0=worst, 1=best)")
    employment_years = st.slider("Employment Years", 0, 40, 5, help="Years of employment")
    age = st.slider("Age", 18, 70, 30, help="Applicant's age")

# Create feature DataFrame in the correct order
features_df = pd.DataFrame({
    'income': [income],
    'loan_amount': [loan_amount],
    'credit_score': [credit_score],
    'employment_years': [employment_years],
    'age': [age],
    'debt_to_income': [debt_to_income],
    'loan_to_income': [loan_to_income]
})

# Scale the features
features_scaled = scaler.transform(features_df)

# Prediction button
if st.button("🔮 Predict Loan Default Risk", type="primary"):
    try:
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Display results
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK**: Loan will likely default")
            st.metric("Default Probability", f"{probability:.1%}")
            
            # Risk factors
            st.warning("**Risk Factors Detected:**")
            if credit_score < 0.3:
                st.write("• Low credit score")
            if debt_to_income > 0.6:
                st.write("• High debt-to-income ratio")
            if loan_to_income > 0.4:
                st.write("• High loan-to-income ratio")
            if employment_years < 2:
                st.write("• Limited employment history")
            if age < 25:
                st.write("• Young applicant")
        else:
            st.success(f"✅ **LOW RISK**: Loan is likely to be repaid")
            st.metric("Default Probability", f"{probability:.1%}")
            
        # Additional insights
        st.info(f"**Model Confidence**: {max(probability, 1-probability):.1%}")
        
        # Risk level
        if probability < 0.3:
            risk_level = "🟢 Low Risk"
        elif probability < 0.7:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🔴 High Risk"
            
        st.write(f"**Risk Level**: {risk_level}")
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

# Add some helpful information
st.markdown("---")
st.subheader("ℹ️ About This Model")
st.write("""
This model uses a Random Forest algorithm trained on synthetic loan data to predict default risk.
The model considers 7 key factors:
- **Financial**: Income, loan amount, debt-to-income ratio, loan-to-income ratio
- **Personal**: Credit score, employment history, age

**Model Performance:**
- Accuracy: ~94%
- ROC-AUC Score: ~99%
- Balanced predictions for both high and low risk cases
""")

# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    st.subheader("📊 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    st.bar_chart(importance_df.set_index('Feature'))
