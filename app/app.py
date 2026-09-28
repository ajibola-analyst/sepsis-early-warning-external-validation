import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import joblib, pandas as pd, streamlit as st
from src.features import build_features

st.title("ICU Sepsis Early-Warning: research demo")
st.caption("Retrospective research prototype. NOT a medical device.")
model = joblib.load("models/lgbm_sepsis.joblib")
meta = json.load(open("models/meta.json"))
up = st.file_uploader("Upload one patient's hourly .psv file (PhysioNet 2019 format)")
if up:
    d = pd.read_csv(up, sep="|"); d["patient_id"] = "upload"
    f = build_features(d)
    f["risk"] = model.predict_proba(f[meta["features"]])[:, 1]
    f["alert"] = f.risk >= meta["threshold"]
    st.line_chart(f.set_index("ICULOS")[["risk"]])
    st.write("Alert threshold:", round(meta["threshold"], 3))
    a = f[f.alert]
    st.write(f"First alert at ICU hour {int(a.ICULOS.min())}" if len(a) else "No alert raised")