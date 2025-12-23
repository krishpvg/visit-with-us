import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="krishpvg/visit-with-us", filename="best_product_taken_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism package Prediction")
st.write("""
Tourism package prediction.
""")

# User input
TypeofContact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
CityTier = int(st.selectbox("City Tier", ["1", "2", "3"]))
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Age = st.number_input("Age",min_value=18, max_value=70, value=18, step=1)
MaritalStatus = st.selectbox("Marital StatusType of contact", ["Single", "Married", "Unmarried", "Divorced"])
Passport = st.checkbox("Passport available?")
Passport = int(Passport)
OwnCar = st.checkbox("Own a car available?")
OwnCar = int(OwnCar)
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("Monthly Income",min_value=1000, max_value=1000000, value=1000, step=1)
PreferredPropertyStar = int(st.selectbox("Preferred Property Star", ["3", "4", "5"]))
NumberOfChildrenVisiting = int(st.selectbox("Number of children visiting", ["0", "1", "2", "3", "4", "5"]))
NumberofTrips = st.number_input("Number of trips",min_value=1, max_value=50, value=1, step=1)
PitchSatisfactionScore = int(st.selectbox("Pitch Satisfaction Score", ["1", "2", "3", "4", "5"]))
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Followups",min_value=1, max_value=10, value=1, step=1)
DurationOfPitch = st.number_input("Duration of Pitch",min_value=1, max_value=100, value=1, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'Age': Age,
    'Numberoftrips': NumberofTrips,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# Predict button
if st.button("Predict Tourism Package Purchase"):
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0, 1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("The customer is likely to purchase the tourism package ✅")
    else:
        st.warning("The customer is unlikely to purchase the tourism package ❌")
    
    st.info(f"Predicted probability of purchase: {prediction_prob*100:.2f}%")
