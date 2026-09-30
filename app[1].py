import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Solar Generation Classifier", page_icon="☀️")

@st.cache_resource
def load_model():
    return joblib.load("models/solar_generation_classifier.joblib")

model = load_model()

st.title("☀️ Solar Generation Condition Classification")
st.write("Educational ML prototype for classifying solar-generation operating conditions.")

st.sidebar.header("Input Parameters")
solar_output = st.sidebar.number_input("Solar Output (kW)", 0.0, 320.0, 180.0)
irradiance = st.sidebar.number_input("Irradiance (W/m²)", 30.0, 1100.0, 700.0)
temperature = st.sidebar.number_input("Temperature (°C)", 15.0, 42.0, 28.0)
cloud = st.sidebar.slider("Cloud Indicator", 0.0, 1.0, 0.30, 0.01)
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening"])

if st.button("Predict Generation Condition"):
    input_df = pd.DataFrame([{
        "solar_output_kW": solar_output,
        "irradiance_W_m2": irradiance,
        "temperature_C": temperature,
        "cloud_indicator": cloud,
        "time_of_day": time_of_day
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Condition: **{prediction}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probabilities})
        st.subheader("Prediction Probability")
        st.dataframe(prob_df, hide_index=True)
        st.bar_chart(prob_df.set_index("Class"))

st.caption("Synthetic educational dataset; not intended for real-world operational decisions.")
