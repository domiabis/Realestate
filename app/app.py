import streamlit as st
import pandas as pd
import pickle
import numpy as np

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    reg_model = pickle.load(open("C:/Realestate/app/models/regression_model.pkl", "rb"))
    cls_model = pickle.load(open("C:/Realestate/app/models/classification_model.pkl", "rb"))
    return reg_model, cls_model

reg_model, cls_model = load_models()

# =========================
# Streamlit App UI
# =========================
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Investment Advisor")
st.write("Provide property details below to get predictions and recommendations.")

# =========================
# Input Form
# =========================
with st.form("input_form"):
    st.subheader("📌 Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        BHK = st.number_input("BHK", 1, 10, 2)
        Size = st.number_input("Size (SqFt)", 300, 10000, 1200)
        Price = st.number_input("Price (Lakhs)", 5.0, 10000.0, 75.0)
        Parking_Space = st.selectbox("Parking Space", ["Yes", "No"])

    with col2:
        Age = st.number_input("Age of Property (Years)", 0, 50, 5)
        Amenities_Count = st.number_input("Amenities Count", 0, 20, 5)
        Nearby_Hospitals = st.number_input("Nearby Hospitals (0-10)", 0, 10, 3)
        Security = st.selectbox("Security", ["Yes", "No"])

    with col3:
        Nearby_Schools = st.number_input("Nearby Schools (0-10)", 0, 10, 4)
        City = st.selectbox("City", [
            'Chennai','Pune','Ludhiana','Jodhpur','Jaipur','Durgapur','Coimbatore',
            'Bilaspur','New Delhi','Ranchi','Warangal','Bangalore','Nagpur',
            'Lucknow','Silchar','Dehradun','Noida','Gaya','Jamshedpur','Ahmedabad',
            'Hyderabad','Faridabad','Amritsar','Kolkata','Dwarka','Vishakhapatnam',
            'Bhopal','Indore','Haridwar','Mysore','Patna','Raipur','Vijayawada',
            'Trivandrum','Kochi','Surat','Gurgaon','Mangalore','Cuttack',
            'Bhubaneswar','Guwahati','Mumbai'
        ])
        Property_Type = st.selectbox("Property Type", ["Apartment", "Villa", "Independent House"])
        Facing = st.selectbox("Facing", ["North", "South", "East", "West"])
    submitted = st.form_submit_button("Predict")

# =========================
# Prediction Logic
# =========================
if submitted:

    # Build input dataframe EXACTLY matching training features
    input_df = pd.DataFrame([{
        "BHK": BHK,
        "Size_in_SqFt": Size,
        "Price_in_Lakhs": Price,
        "Age_of_Property": Age,
        "Amenities_Count": Amenities_Count,
        "Nearby_Hospitals": Nearby_Hospitals,
        "Nearby_Schools": Nearby_Schools,
        "City": City,
        "Property_Type": Property_Type,
        "Facing": Facing,
        "Parking_Space": Parking_Space,
        "Security": Security
    }])

    # Regression Prediction
    predicted_price = reg_model.predict(input_df)[0]

    # Classification Prediction
    invest_pred = cls_model.predict(input_df)[0]
    invest_prob = cls_model.predict_proba(input_df)[0][1] * 100

    # =========================
    # Output Display
    # =========================
    st.success(f"💰 **Estimated Price After 5 Years:** ₹{predicted_price:.2f} Lakhs")

    if invest_pred == 1:
        st.markdown(
            f"<h3 style='color:green;'>✔ Recommended Investment (Confidence: {invest_prob:.2f}%)</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h3 style='color:red;'>❌ Not Recommended (Confidence: {invest_prob:.2f}%)</h3>",
            unsafe_allow_html=True
        )