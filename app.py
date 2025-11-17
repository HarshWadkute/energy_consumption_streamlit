# app.py
import streamlit as st
import joblib, json
from pathlib import Path
import pandas as pd
import numpy as np

st.set_page_config(page_title="Energy Consumption Predictor", layout="centered")
st.title("⚡ Energy Consumption Predictor (Stacking Regressor)")

MODEL_PATH = Path("model/energy_pipeline_stacking.pkl")
CATS_PATH = Path("model/categories.json")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CATS_PATH.exists():
        raise FileNotFoundError(f"Categories file not found: {CATS_PATH}")
    model = joblib.load(MODEL_PATH.open("rb"))
    with open(CATS_PATH, "r") as f:
        categories = json.load(f)
    return model, categories

try:
    model, categories = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model or categories: {e}")
    st.stop()

st.write("Enter building & environment features below. Feature names and types match the training notebook.")

# ---------- INPUTS: MUST match exact column *names* used in training ----------
# Numeric features
col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("Square Footage", min_value=0.0, value=1200.0, step=1.0)
    occupants = st.number_input("Number of Occupants", min_value=0, value=3, step=1)
    appliances = st.number_input("Appliances Used (count)", min_value=0, value=5, step=1)
with col2:
    avg_temp = st.number_input("Average Temperature (°C)", min_value=-50.0, value=24.0, step=0.1)

# Categorical features (populate from categories.json)
building_types = categories.get("Building Type", [])
day_of_week = categories.get("Day of Week", [])

# If categories are empty arrays, show a text input fallback
if building_types:
    building = st.selectbox("Building Type", building_types)
else:
    building = st.text_input("Building Type (free text)")

if day_of_week:
    day = st.selectbox("Day of Week", day_of_week)
else:
    day = st.text_input("Day of Week (free text)")

# Build DataFrame in same column order/names as training
input_df = pd.DataFrame([{
    "Square Footage": float(sqft),
    "Number of Occupants": int(occupants),
    "Appliances Used": int(appliances),
    "Average Temperature": float(avg_temp),
    "Building Type": str(building),
    "Day of Week": str(day)
}])

st.subheader("Input preview")
st.dataframe(input_df)

if st.button("Predict energy consumption"):
    try:
        # pipeline includes preprocessing, so pass DataFrame
        pred = model.predict(input_df)
        val = float(pred[0])
        st.success(f"Predicted energy consumption: **{val:.3f}** (same units used in training)")
        # If model supports predict_proba or other diagnostics, show them here.
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("If you saved only the raw model (without preprocessing), you must apply the same preprocessing steps before calling predict.")
