import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Churn Prediction",
    layout="centered"
)


st.title("Churn Prediction")

st.write("Prediction of churn customers.")


model = joblib.load("data/processed/models/final_model.pkl")

# print(model)

st.header("Customer Informations")


gender = st.selectbox(
    "gender",
    ["Male", "Female"]
)


tenure = st.number_input(
    "tenure",
    max_value=100
)


internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL", "Fibre optic", "No"
    ]
)



