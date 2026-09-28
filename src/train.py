import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_curve
from src.data import load_set
from src.features import build_features, feature_columns

RAW, PROC = Path("data/raw"), Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def get(name, folder):
    f = PROC / f"{name}.parquet"
    if f.exists(): return pd.read_parquet(f)
    df = build_features(load_set(folder, name)); df.to_parquet(f); return df

A = get("A", RAW / "training_setA")
B = get("B", RAW / "training_setB")
feats = feature_columns(A)

# Split Set A BY PATIENT (never by row): 70% train / 15% validation / 15% internal test
g1 = GroupShuffleSplit(1, test_size=0.30, random_state=42)
tr_i, rest_i = next(g1.split(A, groups=A.patient_id))
rest = A.iloc[rest_i]
g2 = GroupShuffleSplit(1, test_size=0.50, random_state=42)
va_i, te_i = next(g2.split(rest, groups=rest.patient_id))
tr, va, teA = A.iloc[tr_i], rest.iloc[va_i], rest.iloc[te_i]

model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           random_state=42, n_jobs=-1)
model.fit(tr[feats], tr.SepsisLabel, eval_set=[(va[feats], va.SepsisLabel)],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

# Alert threshold chosen on VALIDATION only (max F1); never tuned on Set B
p_va = model.predict_proba(va[feats])[:, 1]
pr, rc, th = precision_recall_curve(va.SepsisLabel, p_va)
f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-9)
thr = float(th[np.argmax(f1)])

joblib.dump(model, "models/lgbm_sepsis.joblib")
json.dump({"threshold": thr, "features": feats}, open("models/meta.json", "w"))

def pred(d, name):
    o = d[["patient_id", "ICULOS", "SepsisLabel", "Age", "Gender", "shock_index"]].copy()
    o["p"] = model.predict_proba(d[feats])[:, 1]; o["split"] = name
    return o
pd.concat([pred(teA, "testA"), pred(B, "B")]).reset_index(drop=True).to_parquet("reports/predictions.parquet")
print("Done. Threshold:", thr)