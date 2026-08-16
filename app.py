import streamlit as st
import joblib
import pandas as pd
import numpy as np
st.title("Life Expectancy Prediction")
year = st.number_input("Year", min_value=2000, max_value=2030, value=2015)

status = st.selectbox("Status", ["Developed", "Developing"])

adult_mortality = st.number_input("Adult Mortality", value=150.0)

infant_deaths = st.number_input("Infant Deaths", value=10)

alcohol = st.number_input("Alcohol", value=4.0)

percentage_expenditure = st.number_input("Percentage Expenditure", value=100.0)

hepatitis_b = st.number_input("Hepatitis B", value=80.0)

measles = st.number_input("Measles", value=0)

bmi = st.number_input("BMI", value=25.0)

under_five_deaths = st.number_input("Under-five Deaths", value=10)

polio = st.number_input("Polio", value=85.0)

total_expenditure = st.number_input("Total Expenditure", value=6.0)

diphtheria = st.number_input("Diphtheria", value=85.0)

hiv_aids = st.number_input("HIV/AIDS", value=0.1)

gdp = st.number_input("GDP", value=5000.0)

population = st.number_input("Population", value=1000000.0)

thinness_1_19 = st.number_input("Thinness 1-19 Years", value=5.0)

thinness_5_9 = st.number_input("Thinness 5-9 Years", value=5.0)

income_composition = st.number_input("Income Composition of Resources", value=0.70)

schooling = st.number_input("Schooling", value=12.0)

# Encode Status
status_encoded = 0 if status == "Developed" else 1
input_data = pd.DataFrame({
    "Year": [year],
    "Status": [status_encoded],
    "Adult Mortality": [adult_mortality],
    "infant deaths": [infant_deaths],
    "Alcohol": [alcohol],
    "percentage expenditure": [percentage_expenditure],
    "Hepatitis B": [hepatitis_b],
    "Measles": [measles],
    "BMI": [bmi],
    "under-five deaths": [under_five_deaths],
    "Polio": [polio],
    "Total expenditure": [total_expenditure],
    "Diphtheria": [diphtheria],
    "HIV/AIDS": [hiv_aids],
    "GDP": [gdp],
    "Population": [population],
    "thinness 1-19 years": [thinness_1_19],
    "thinness 5-9 years": [thinness_5_9],
    "Income composition of resources": [income_composition],
    "Schooling": [schooling]
})


model = joblib.load("random_forest_model.pkl")

if st.button("Predict Life Expectancy"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Life Expectancy: {prediction[0]:.2f} years"
    )