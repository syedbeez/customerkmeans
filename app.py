import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load saved model and scaler
with open("kmeans_model.pkl", "rb") as file:
    kmeans = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load dataset for country encoding reference
df = pd.read_excel("OnlineRetail.xlsx")

# Encode Country dynamically
encoder = LabelEncoder()
df["Country"] = encoder.fit_transform(df["Country"])
country_mapping = dict(zip(df["Country"], encoder.classes_))

st.title("Customer Segmentation using K-Means")

st.markdown("This app predicts which customer segment a user belongs to based on their purchase behavior.")

# Input fields
quantity = st.number_input("Quantity", min_value=1, step=1, help="Total items purchased")
unit_price = st.number_input("Unit Price", min_value=0.01, step=0.01, help="Price per unit")
country = st.text_input("Country", help="Enter the customer's country")

if st.button("Predict Cluster"):
    if country in country_mapping.values():
        country_encoded = list(country_mapping.keys())[list(country_mapping.values()).index(country)]
        total_price = quantity * unit_price
        features = np.array([[quantity, unit_price, total_price, country_encoded]])
        scaled_features = scaler.transform(features)
        cluster = kmeans.predict(scaled_features)[0]
        st.success(f"The customer belongs to Cluster {cluster}")
    else:
        st.error("Country not found in the dataset. Try a different one.")
