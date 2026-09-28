import json, joblib, pandas as pd, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
model = joblib.load("models/lgbm_sepsis.joblib")
feats = json.load(open("models/meta.json"))["features"]
X = pd.read_parquet("data/processed/B.parquet")[feats].sample(20000, random_state=0)
sv = shap.TreeExplainer(model).shap_values(X)
sv = sv[1] if isinstance(sv, list) else sv
shap.summary_plot(sv, X, max_display=20, show=False); plt.tight_layout()
plt.savefig("reports/figures/shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
shap.dependence_plot("shock_index", sv, X, show=False)
plt.savefig("reports/figures/shap_shock_index.png", dpi=200, bbox_inches="tight")