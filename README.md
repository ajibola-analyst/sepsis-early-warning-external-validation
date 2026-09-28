# Sepsis Early Warning: External Validation Across Hospital Systems

**Live Demo:** [Streamlit App](https://sepsis-early-warning-external-validation.streamlit.app/) | **DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23019969.svg)](https://doi.org/10.5281/zenodo.23019969) | **Preprint:** *In Progress*

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![License](https://img.shields.io/badge/Code-MIT-lightgrey)

> Research prototype. Not a medical device. Not for clinical use.

## Summary
Sepsis kills more patients when it is recognised late. This project develops a
physiology-informed machine-learning model that flags ICU patients up to 6 hours before
clinical sepsis onset, and tests whether it still works in a different hospital system.
The model was developed on one hospital system (Set A) and evaluated once, without
retuning, on another (Set B). Performance, calibration, subgroup behaviour and the
physiological plausibility of its predictions are all reported.

Author: Ajibola Odeyemi, Specialist in Quantitative & Qualitative Analytics. Physiology training guided the feature
design and the interpretation of the model.

## Research question
Can routinely collected ICU vital signs and laboratory values, engineered using
physiological reasoning, predict sepsis several hours before onset, and do the results
hold across hospital systems?

## Data
PhysioNet/Computing in Cardiology Challenge 2019, "Early Prediction of Sepsis From
Clinical Data" (Reyna et al., Crit Care Med 2020), accessed via a Kaggle mirror.
- Set A: 20,336 patients (development). Set B: 20,000 patients from a different hospital
  system (external validation only).
- 40 hourly variables (vitals, labs, demographics) plus `SepsisLabel`.
- Licence: CC BY-NC-SA 4.0. Data are not redistributed here; see `data/README.md`.

## Methods
| Step | Detail |
|---|---|
| Split | Set A split by patient (70/15/15 train/validation/internal test); never by row |
| Features | Forward-filled values, "measured" indicators, shock index, pulse pressure, 6-hour rolling mean/SD/change of key vitals, SOFA-inspired organ flags |
| Leakage control | Features use only data up to the current hour; verified by unit test |
| Model | LightGBM with early stopping on the validation set |
| Threshold | Chosen on validation data only (maximum F1) |
| External test | Set B evaluated once, no retuning |
| Uncertainty | Patient-level bootstrap 95% confidence intervals |
| Extra analyses | Calibration, subgroup (sex, age), SHAP interpretation |

## Results
| Metric | Set A internal test | Set B external |
|---|---|---|
| AUROC (95% CI) | 0.828 (0.806-0.851) | 0.776 (0.762-0.789) |
| AUPRC (95% CI) | 0.119 (0.102-0.144) | 0.071 (0.064-0.079) |
| Shock index alone, AUROC | 0.592 | 0.607 |
| Brier score | 0.0207 | 0.0142 |

At the validation-chosen threshold on Set B, 42.8% of sepsis patients (n=1,142) triggered an alert before onset, and 6.8% of non-sepsis patients triggered at least one alert.

Subgroup AUROC on Set B (sex and age bands): see `reports/figures/subgroups.png`.

![ROC](reports/figures/roc.png)
![Calibration](reports/figures/calibration.png)
![SHAP](reports/figures/shap_summary.png)

## Physiological interpretation
Model predictions are strongly driven by systemic markers of tissue perfusion, cellular dysfunction, and acute inflammatory response[cite: 5]. High `Lactate` (>2.0 mmol/L) and elevated `FiO2` (increased oxygen demand) serve as primary indicators of anaerobic metabolism and respiratory failure[cite: 5]. `Resp_mean6` (tachypnea) and elevated `WBC` (leukocytosis) capture early systemic inflammatory response syndrome (SIRS)[cite: 5]. Renal markers (`Creatinine`, `BUN`) reflect progressive acute kidney injury secondary to septic hypoperfusion[cite: 5]. Notably, SHAP analysis of `shock_index` (Heart Rate / Systolic Blood Pressure) demonstrates a sharp non-linear increase in predicted sepsis risk beyond 0.9–1.0, capturing compensated circulatory collapse prior to overt hypotension. Non-physiological features like `ICULOS` (ICU length of stay) and `HospAdmTime` also contribute significantly, reflecting time-dependent baseline disease severity and clinical workflow patterns[cite: 5].

## Limitations
- Retrospective data from two US hospital systems; not validated in African settings.
- Labels follow a Sepsis-3-derived definition with a 6-hour pre-onset window.
- Measurement patterns reflect clinician behaviour and may not transfer between sites.
- Metrics are row-level. The official Challenge utility score is not reported.
- Not validated on MIMIC-IV or eICU, and not tested prospectively.
- Calibration on Set B shows over-prediction at high predicted risk; recalibration would be needed before any use in a new setting.

## Repository structure
```text
sepsis-early-warning/
├── app/app.py            Streamlit demo
├── data/README.md        Data download instructions (data not included)
├── models/               Trained model and metadata
├── reports/              metrics.json and figures
├── src/                  data, features, train, evaluate, explain
├── tests/                Unit tests (feature correctness, no future leakage)
├── CITATION.cff
├── requirements.txt
└── README.md
```

## Reproduce
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Place training_setA and training_setB in data/raw/ (see data/README.md)
pytest
python -m src.train
python -m src.evaluate
python -m src.explain
streamlit run app/app.py
```

## Future work
Sequence models (GRU/LSTM), validation on MIMIC-IV/eICU, and validation on African
ICU cohorts.

## Citation and licence
Please cite the dataset (Reyna MA et al., Crit Care Med 2020) and this repository
(see `CITATION.cff`). Code: MIT. Data: CC BY-NC-SA 4.0 (original authors).