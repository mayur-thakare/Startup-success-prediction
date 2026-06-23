import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set a professional page configuration
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

# 1. Load the trained model and dataset to extract valid encoder mappings dynamically
@st.cache_resource
def load_artifacts():
    # Load your exact saved Random Forest model
    with open('startup_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load your CSV data to fit encoders matching your notebook setup exactly
    df = pd.read_csv('indian_startup_funding.csv')
    
    # Create the encoder dictionary matching your categorical variables
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in ['Industry', 'SubVertical', 'City']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        encoders[col] = le
        
    return model, encoders, df

try:
    model, encoders, original_df = load_artifacts()
except FileNotFoundError as e:
    st.error("⚠️ Necessary files missing! Make sure 'startup_model.pkl' and 'indian_startup_funding.csv' are in this directory.")
    st.stop()

# Helper function to safely handle inputs and unseen categories
def encode_val(column_name, val):
    le = encoders[column_name]
    # Check if value belongs to known classes; fallback to 0 if unknown
    if val in le.classes_:
        return int(le.transform([val])[0])
    else:
        return 0

# --- App Layout & Design ---
st.title("🚀 Startup Success Predictor")
st.markdown("Provide the funding, region, and structural attributes below to evaluate whether the startup is likely to succeed.")
st.write("---")

# Organized layout columns
col1, col2 = st.columns(2)

with col1:
    # Populates options strictly using clean values from your dataset
    industry = st.selectbox(
        "Industry Sector",
        options=sorted(original_df['Industry'].unique().tolist())
    )
    
    city = st.selectbox(
        "Headquarters City",
        options=sorted(original_df['City'].unique().tolist())
    )
    
    year = st.number_input("Funding Year", min_value=2010, max_value=2030, value=2026)

with col2:
    # Auto-complete suggestions for Sub-Vertical based on selected Industry
    sub_choices = sorted(original_df[original_df['Industry'] == industry]['SubVertical'].unique().tolist())
    subvertical = st.selectbox("Sub-Vertical/Niche", options=sub_choices if sub_choices else ["K12"])
    
    month = st.slider("Funding Month", min_value=1, max_value=12, value=6)
    
    investor_count = st.number_input("Investor Count", min_value=1, max_value=50, value=1)

investment = st.number_input("Investment Amount (USD)", min_value=0.0, value=100000.0, step=1000.0)

st.write("---")

# --- Prediction Engine ---
if st.button("Predict Success Framework", type="primary", use_container_width=True):
    with st.spinner("Processing framework algorithms..."):
        
        # Encode inputs exactly as done in your Jupyter Notebook training sequence
        enc_industry = encode_val('Industry', industry)
        enc_subvertical = encode_val('SubVertical', subvertical)
        enc_city = encode_val('City', city)
        
        # Create user DataFrame matching the exact feature order of your saved model
        # Column names: ['Industry', 'SubVertical', 'City', 'InvestmentAmount_USD', 'year', 'month', 'Investor_Count']
        user_data = pd.DataFrame({
            'Industry': [enc_industry],
            'SubVertical': [enc_subvertical],
            'City': [enc_city],
            'InvestmentAmount_USD': [investment],
            'year': [year],
            'month': [month],
            'Investor_Count': [investor_count]
        })
        
        # Predict class & confidence probabilities
        prediction = model.predict(user_data)[0]
        probabilities = model.predict_proba(user_data)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        # Present clean output banners mirroring your cell predictions
        if prediction == 1:
            st.success("### 🎉 Startup is likely to be Successful")
            st.metric(label="Success Likelihood Confidence", value=f"{confidence * 100:.2f}%")
        else:
            st.error("### ⚠️ Startup is likely to be Unsuccessful")
            st.metric(label="Risk/Failure Probability Confidence", value=f"{confidence * 100:.2f}%")