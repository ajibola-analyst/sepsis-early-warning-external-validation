from pathlib import Path
import pandas as pd

def load_set(folder, hospital):
    """Read every patient .psv in a folder into one DataFrame."""
    frames = []
    for f in sorted(Path(folder).rglob("*.psv")):
        d = pd.read_csv(f, sep="|")
        d["patient_id"] = f"{hospital}_{f.stem}"
        d["hospital"] = hospital
        frames.append(d)
    return pd.concat(frames, ignore_index=True)