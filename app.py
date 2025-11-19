import streamlit as st
import joblib, json
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Energy Consumption Predictor", layout="centered")
st.title("⚡ Energy Consumption Predictor (LabelEncoder version)")

MODEL_DIR = Path("model")
MODEL = MODEL_DIR / "stacking_regressor_only.pkl"
SCALER = MODEL_DIR / "standard_scaler.pkl"
LE_B = MODEL_DIR / "le_building.pkl"
LE_D = MODEL_DIR / "le_day.pkl"
META = MODEL_DIR / "model_meta.json"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL.open("rb"))
    scaler = joblib.load(SCALER.open("rb"))
    le_building = joblib.load(LE_B.open("rb"))
    le_day = joblib.load(LE_D.open("rb"))
    with open(META) as f:
        meta = json.load(f)
    return model, scaler, le_building, le_day, meta

model, scaler, le_building, le_day, meta = load_artifacts()

numeric_cols = meta["numeric_cols"]
categorical_cols = meta["categorical_cols"]
categories = meta["categories"]

# UI
col1, col2 = st.columns(2)
with col1:
    sf = st.number_input("Square Footage", value=1000.0)
    occ = st.number_input("Number of Occupants", value=2)
    apps = st.number_input("Appliances Used", value=4)
with col2:
    temp = st.number_input("Average Temperature", value=24.0)

building = st.selectbox("Building Type", categories["Building Type"])
day = st.selectbox("Day of Week", categories["Day of Week"])

# Build DF
df_in = pd.DataFrame([{
    "Square Footage": sf,
    "Number of Occupants": occ,
    "Appliances Used": apps,
    "Average Temperature": temp,
    "Building Type": building,
    "Day of Week": day
}])

st.write("Input:", df_in)

if st.button("Predict"):
    try:
        # Numeric
        X_num = scaler.transform(df_in[numeric_cols])

        # Categorical using LabelEncoder
        b_enc = le_building.transform(df_in["Building Type"])
        d_enc = le_day.transform(df_in["Day of Week"])

        X_cat = np.column_stack([b_enc, d_enc])

        # Final feature vector (must match training)
        X_final = np.hstack([X_num, X_cat])

        pred = model.predict(X_final)
        st.success(f"Predicted Energy Consumption: **{float(pred[0]):.3f}**")
    except Exception as e:
        st.error(str(e))
