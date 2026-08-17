import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Electricity Consumption Predictor", page_icon="⚡")

st.title("⚡ Electricity Consumption Predictor")
st.write("Predict the expected electricity consumption using a trained Machine Learning model.")

model = joblib.load("electricity_model.pkl")

st.sidebar.header("Input Parameters")

hour = st.sidebar.slider("Hour of Day", 0, 23, 19)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
is_weekend = 1 if day_of_week >= 5 else 0
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 28.0)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 60)
previous = st.sidebar.number_input("Previous Consumption (kWh)", 0.0, 20.0, 2.5)

input_data = pd.DataFrame([{
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "temperature_C": temperature,
    "humidity_percent": humidity,
    "previous_consumption_kWh": previous
}])

if st.button("Predict Consumption"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Electricity Consumption: {prediction:.2f} kWh")

st.markdown("---")
st.subheader("Project Flow")
st.write("Input Data → Preprocessing → Random Forest Model → Prediction")
