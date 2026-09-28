import numpy as np, pandas as pd
from pandas.testing import assert_frame_equal
from src.features import build_features, CLIN

def toy(n=20):
    r = np.random.default_rng(0)
    d = pd.DataFrame(r.uniform(40, 120, (n, len(CLIN))), columns=CLIN)
    d["Age"], d["Gender"], d["HospAdmTime"] = 60, 1, -5
    d["ICULOS"] = range(1, n + 1); d["patient_id"] = "x"; d["hospital"] = "A"; d["SepsisLabel"] = 0
    return d

def test_shock_index():
    f = build_features(toy())
    assert np.allclose(f.shock_index, f.HR / f.SBP)

def test_no_future_leakage():
    d = toy(); d2 = d.copy(); d2.loc[15:, "HR"] = 999
    assert_frame_equal(build_features(d).iloc[:15], build_features(d2).iloc[:15])