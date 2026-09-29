import numpy as np
import pandas as pd

CLIN = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2",
        "pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine",
        "Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium",
        "Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
DEMO = ["Age", "Gender", "HospAdmTime", "ICULOS"]  
ROLL = ["HR","MAP","SBP","Resp","Temp","O2Sat","shock_index"]
META = ("patient_id", "hospital", "SepsisLabel")

def build_features(df):
    """Causal features only: value at hour t uses data from hours <= t."""
    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    pid = df["patient_id"]
    ff = df[CLIN].groupby(pid).ffill()          
    out = pd.DataFrame(index=df.index)
    for c in META:
        if c in df: out[c] = df[c]
    for c in DEMO: out[c] = df[c]
    for c in CLIN:
        out[c] = ff[c]
        out[c + "_measured"] = df[c].notna().astype("int8")   
    out["shock_index"] = (ff.HR / ff.SBP).replace([np.inf, -np.inf], np.nan)
    out["pulse_pressure"] = ff.SBP - ff.DBP
    ffx = ff.assign(shock_index=out["shock_index"])
    for v in ROLL:
        gb = ffx[v].groupby(pid)
        out[f"{v}_mean6"] = gb.rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{v}_std6"] = gb.rolling(6, min_periods=2).std().reset_index(level=0, drop=True)
        out[f"{v}_delta6"] = ffx[v] - gb.shift(6)          
    # SOFA-INSPIRED organ-dysfunction flags (not a formal SOFA score)
    out["flag_map_low"] = (ff.MAP < 70).astype("int8")
    out["flag_lactate_high"] = (ff.Lactate > 18).astype("int8")   
    out["flag_creat_high"] = (ff.Creatinine >= 2).astype("int8")
    out["flag_bili_high"] = (ff.Bilirubin_total >= 2).astype("int8")
    out["flag_plt_low"] = (ff.Platelets < 100).astype("int8")
    out["organ_flags"] = out[[c for c in out if c.startswith("flag_")]].sum(axis=1).astype("int8")
    f64 = out.select_dtypes("float64").columns
    out[f64] = out[f64].astype("float32")
    return out

def feature_columns(df):
    return [c for c in df.columns if c not in META]