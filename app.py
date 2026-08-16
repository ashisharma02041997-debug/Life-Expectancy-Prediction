import streamlit as st
import pandas as pd
import joblib


# ==================================================
# LOAD TRAINED FILES
# ==================================================

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


# ==================================================
# PAGE TITLE
# ==================================================

st.title("Life Expectancy Prediction")

st.write(
    "Enter the health, economic and demographic details below "
    "to predict Life Expectancy."
)


# ==================================================
# USER INPUTS
# ==================================================

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2030,
    value=2015
)

status = st.selectbox(
    "Status",
    ["Developed", "Developing"]
)

adult_mortality = st.number_input(
    "Adult Mortality",
    min_value=0.0,
    value=150.0
)

infant_deaths = st.number_input(
    "Infant Deaths",
    min_value=0,
    value=10
)

alcohol = st.number_input(
    "Alcohol",
    min_value=0.0,
    value=4.0
)

percentage_expenditure = st.number_input(
    "Percentage Expenditure",
    min_value=0.0,
    value=100.0
)

hepatitis_b = st.number_input(
    "Hepatitis B",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

measles = st.number_input(
    "Measles",
    min_value=0,
    value=0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

under_five_deaths = st.number_input(
    "Under-five Deaths",
    min_value=0,
    value=10
)

polio = st.number_input(
    "Polio",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

total_expenditure = st.number_input(
    "Total Expenditure",
    min_value=0.0,
    value=6.0
)

diphtheria = st.number_input(
    "Diphtheria",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

hiv_aids = st.number_input(
    "HIV/AIDS",
    min_value=0.0,
    value=0.1
)

gdp = st.number_input(
    "GDP",
    min_value=0.0,
    value=5000.0
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000000.0
)

thinness_1_19 = st.number_input(
    "Thinness 1-19 Years",
    min_value=0.0,
    value=5.0
)

thinness_5_9 = st.number_input(
    "Thinness 5-9 Years",
    min_value=0.0,
    value=5.0
)

income_composition = st.number_input(
    "Income Composition of Resources",
    min_value=0.0,
    max_value=1.0,
    value=0.70
)

schooling = st.number_input(
    "Schooling",
    min_value=0.0,
    value=12.0
)


# ==================================================
# ENCODE STATUS
# ==================================================

# LabelEncoder mapping:
# Developed = 0
# Developing = 1

status_encoded = 0 if status == "Developed" else 1


# ==================================================
# CREATE INPUT DATAFRAME
# ==================================================

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


# ==================================================
# MATCH SAME FEATURE ORDER USED DURING TRAINING
# ==================================================

input_data = input_data[feature_names]


# ==================================================
# SCALE INPUT DATA
# ==================================================

input_scaled = scaler.transform(input_data)


# ==================================================
# PREDICT LIFE EXPECTANCY
# ==================================================

if st.button("Predict Life Expectancy"):

    prediction = model.predict(input_scaled)

    predicted_value = prediction[0]

    st.success(
        f"Predicted Life Expectancy: {predicted_value:.2f} years"
    )


# ==================================================
# MODEL INFORMATION
# ==================================================

st.markdown("---")

st.subheader("Model Information")

st.write("Machine Learning Type: Supervised Machine Learning")
st.write("Problem Type: Regression")
st.write("Final Model: Random Forest Regressor")
st.write("Target Variable: Life Expectancy")