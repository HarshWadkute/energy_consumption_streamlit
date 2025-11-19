# app.py
import streamlit as st
import joblib, json, os
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Energy Consumption Predictor", layout="centered")
st.title("⚡ Energy Consumption Predictor — result in kWh")

# ---------- Files ----------
MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "stacking_regressor_only.pkl"
SCALER_FILE = MODEL_DIR / "standard_scaler.pkl"
LE_BUILDING_FILE = MODEL_DIR / "le_building.pkl"
LE_DAY_FILE = MODEL_DIR / "le_day.pkl"
META_FILE = MODEL_DIR / "model_meta.json"
TARGET_SCALER_FILE = MODEL_DIR / "target_scaler.pkl"  # should be created from notebook

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    # require core artifacts
    for p in (MODEL_FILE, SCALER_FILE, LE_BUILDING_FILE, LE_DAY_FILE, META_FILE):
        if not p.exists():
            raise FileNotFoundError(f"Missing file in model/: {p.name}")
    model = joblib.load(MODEL_FILE.open("rb"))
    scaler = joblib.load(SCALER_FILE.open("rb"))
    le_building = joblib.load(LE_BUILDING_FILE.open("rb"))
    le_day = joblib.load(LE_DAY_FILE.open("rb"))
    meta = json.load(open(META_FILE, "r"))
    target_scaler = None
    if TARGET_SCALER_FILE.exists():
        target_scaler = joblib.load(TARGET_SCALER_FILE.open("rb"))
    return model, scaler, le_building, le_day, meta, target_scaler

try:
    model, scaler, le_building, le_day, meta, target_scaler = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ---------- Friendly label mapping (edit RHS if you prefer other strings) ----------
code_to_label_building = {
    "0": "Residential",
    "1": "Commercial",
    "2": "Industrial"
}
code_to_label_day = {
    "0": "Weekday",
    "1": "Weekend"
}
label_to_code_building = {v: k for k, v in code_to_label_building.items()}
label_to_code_day = {v: k for k, v in code_to_label_day.items()}

def cast_for_encoder(code_string, encoder):
    classes = list(encoder.classes_)
    if len(classes) == 0:
        return code_string
    cls0 = classes[0]
    # numeric class types -> int
    if isinstance(cls0, (int, np.integer)):
        try:
            return int(code_string)
        except:
            return code_string
    return str(code_string)

# ---------- Inputs ----------
numeric_cols = meta.get("numeric_cols", ["Square Footage","Number of Occupants","Appliances Used","Average Temperature"])
categorical_cols = meta.get("categorical_cols", ["Building Type","Day of Week"])
feature_cols = meta.get("feature_cols", numeric_cols + categorical_cols)

col1, col2 = st.columns(2)
with col1:
    sf = st.number_input(numeric_cols[0], value=1200.0, step=1.0)
    occ = st.number_input(numeric_cols[1], value=3, step=1)
    apps = st.number_input(numeric_cols[2], value=5, step=1)
with col2:
    temp = st.number_input(numeric_cols[3], value=24.0, step=0.1)

building_label = st.selectbox("Building Type", list(code_to_label_building.values()))
day_label = st.selectbox("Day of Week", list(code_to_label_day.values()))

selected_building_code = label_to_code_building[building_label]
selected_day_code = label_to_code_day[day_label]
encoder_input_building = cast_for_encoder(selected_building_code, le_building)
encoder_input_day = cast_for_encoder(selected_day_code, le_day)

# Build input DF in the expected order
input_dict = {
    numeric_cols[0]: float(sf),
    numeric_cols[1]: int(occ),
    numeric_cols[2]: int(apps),
    numeric_cols[3]: float(temp),
    categorical_cols[0]: encoder_input_building,
    categorical_cols[1]: encoder_input_day
}
# Use the exact order model expects if available
if hasattr(model, "feature_names_in_"):
    feature_order = list(model.feature_names_in_)
else:
    feature_order = feature_cols

input_df = pd.DataFrame([input_dict], columns=feature_order)
st.subheader("Input preview (sent to preprocessing)")
st.dataframe(input_df)

# ---------- Predict (with inverse-scaling to kWh if available) ----------
if st.button("Predict (kWh)"):
    try:
        # Numeric scaling (assume scaler expects numeric_cols order)
        X_num = input_df[numeric_cols].values
        X_num_scaled = scaler.transform(X_num)

        # Categorical encoded
        b_enc = le_building.transform([encoder_input_building]).reshape(-1,1)
        d_enc = le_day.transform([encoder_input_day]).reshape(-1,1)
        X_cat_enc = np.hstack([b_enc, d_enc])

        # Build final X in expected order
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            mapping = {}
            # Fill mapping from numeric scaled and categorical enc
            for i, col in enumerate(numeric_cols):
                mapping[col] = X_num_scaled[0, i]
            mapping[categorical_cols[0]] = int(b_enc[0,0])
            mapping[categorical_cols[1]] = int(d_enc[0,0])
            final_row = [mapping[col] for col in expected]
            X_final = np.array(final_row).reshape(1, -1)
        else:
            X_final = np.hstack([X_num_scaled, X_cat_enc])

        st.write("Final feature vector sent to model:")
        st.write(X_final)

        # Predict
        pred_raw = model.predict(X_final).reshape(-1,1)  # shape (1,1)

        # If target scaler present, inverse-transform to original kWh units
        if target_scaler is not None:
            try:
                pred_kwh = target_scaler.inverse_transform(pred_raw).reshape(-1)[0]
                st.success(f"Predicted energy consumption: **{pred_kwh:.2f} kWh**")
                st.caption("Units: kWh (inverse-transformed using saved target scaler).")
            except Exception as e:
                st.warning(f"Target scaler present but inverse_transform failed: {e}")
                st.success(f"Predicted (raw model output): {float(pred_raw[0,0]):.6f} (likely scaled)")
        else:
            # If no scaler, warn user this may be scaled; ask them to save target_scaler
            st.warning("No target scaler found. If your model was trained on scaled target values, the raw output may be in scaled units.")
            st.success(f"Predicted (raw model output): {float(pred_raw[0,0]):.6f} (units unknown)")
            st.info("To show results in kWh, save the target scaler as 'model/target_scaler.pkl' from the training notebook and redeploy.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

# Footer: show notebook path used to create artifacts (local)
st.markdown("---")
st.markdown("Notebook (local path used during preparation):")
st.code("/mnt/data/Energy_consumption_Main_(2).ipynb")
