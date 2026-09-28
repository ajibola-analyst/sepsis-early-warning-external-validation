import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
P = pd.read_parquet("reports/predictions.parquet")
thr = json.load(open("models/meta.json"))["threshold"]
S = {"A_internal_test": P[P.split == "testA"].reset_index(drop=True),
     "B_external": P[P.split == "B"].reset_index(drop=True)}

def boot(d, n=100, seed=0):
    """Patient-level bootstrap (resample patients, not rows) for 95% CIs."""
    rng = np.random.default_rng(seed)
    idx = d.groupby("patient_id").indices
    arrs = list(idx.values()); y, p = d.SepsisLabel.values, d.p.values
    au, ap = [], []
    for _ in range(n):
        ii = np.concatenate([arrs[i] for i in rng.integers(0, len(arrs), len(arrs))])
        au.append(roc_auc_score(y[ii], p[ii])); ap.append(average_precision_score(y[ii], p[ii]))
    return np.percentile(au, [2.5, 97.5]).round(4).tolist(), np.percentile(ap, [2.5, 97.5]).round(4).tolist()

def alerts(d):
    d = d.assign(alert=d.p >= thr)
    first = d[d.alert].groupby("patient_id").ICULOS.min()
    onset = d[d.SepsisLabel == 1].groupby("patient_id").ICULOS.min() + 6   # label starts 6h before onset
    nons = np.setdiff1d(d.patient_id.unique(), onset.index.values)
    lead = (onset - first.reindex(onset.index)).dropna()
    early = lead[lead >= 0]
    return {"sepsis_patients": int(len(onset)),
            "pct_sepsis_alerted_before_onset": round(100 * len(early) / len(onset), 1),
            "median_lead_time_hours": float(early.median()) if len(early) else None,
            "pct_nonsepsis_patients_with_any_alert": round(100 * first.index.isin(nons).sum() / len(nons), 1)}

R = {"threshold": thr}
for name, d in S.items():
    y, p = d.SepsisLabel, d.p
    ci_a, ci_p = boot(d)
    R[name] = {"AUROC": round(roc_auc_score(y, p), 4), "AUROC_95CI": ci_a,
               "AUPRC": round(average_precision_score(y, p), 4), "AUPRC_95CI": ci_p,
               "prevalence_rows": round(float(y.mean()), 4), "Brier": round(brier_score_loss(y, p), 4),
               "baseline_shock_index_AUROC": round(roc_auc_score(y[d.shock_index.notna()], d.shock_index.dropna()), 4),
               "patient_level": alerts(d)}

B = S["B_external"]
sub = {}
for lab, m in {"Female": B.Gender == 0, "Male": B.Gender == 1, "Age<50": B.Age < 50,
               "Age 50-65": (B.Age >= 50) & (B.Age < 65), "Age 65-80": (B.Age >= 65) & (B.Age < 80),
               "Age 80+": B.Age >= 80}.items():
    sub[lab] = round(roc_auc_score(B.SepsisLabel[m], B.p[m]), 4)
R["B_subgroup_AUROC"] = sub
json.dump(R, open("reports/metrics.json", "w"), indent=2)
print(json.dumps(R, indent=2))

plt.figure(figsize=(5, 5))
for name, d in S.items():
    f, t, _ = roc_curve(d.SepsisLabel, d.p); plt.plot(f, t, label=f"{name} (AUROC {R[name]['AUROC']})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.legend(); plt.title("ROC: internal vs external hospital"); plt.tight_layout(); plt.savefig(FIG / "roc.png", dpi=200); plt.close()

fr, mp = calibration_curve(B.SepsisLabel, B.p, n_bins=10, strategy="quantile")
plt.figure(figsize=(5, 5)); plt.plot(mp, fr, "o-"); plt.plot([0, mp.max()], [0, mp.max()], "k--")
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.title("Calibration on external Set B")
plt.tight_layout(); plt.savefig(FIG / "calibration.png", dpi=200); plt.close()

plt.figure(figsize=(7, 4)); plt.bar(sub.keys(), sub.values()); plt.ylim(0.5, 1); plt.ylabel("AUROC")
plt.xticks(rotation=30); plt.title("External performance by subgroup"); plt.tight_layout()
plt.savefig(FIG / "subgroups.png", dpi=200)