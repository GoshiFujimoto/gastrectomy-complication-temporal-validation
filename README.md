# Sequential temporal validation of models predicting complications after gastrectomy in patients aged ≥75 years

Analysis code for: *Sequential Temporal Validation of Regression and Machine Learning Models for Predicting Complications After Gastrectomy in Patients Aged 75 Years or Older* (Digestive Surgery).
**No patient-level data are included.** Patient data are held at Kameda Medical Center and are available on reasonable request (see the manuscript).

## Design implemented
- Cohort: gastrectomy for gastric cancer, age ≥75 years, April 2000–March 2025 (`lib.load_cohort`).
- Expanding-window temporal validation: training on all cases before each test block; test blocks 2010/4–2014/3, 2014/4–2019/3, 2019/4–2025/3 (`dbl.PRIMARY`).
- Primary model: ridge logistic regression (12 predictors); comparators: CatBoost (fixed hyperparameters), CCI-only and ASA-PS-only logistic regression.
- Primary measures: pooled AUC and calibration slope; incremental value by ΔAUC (DeLong, Holm for 2 tests), ΔBrier, decision curve analysis; NRI/IDI secondary.
- Exploratory: evaluation by surgical approach, approach-specific recalibration, sensitivity analyses, standardized coefficients.

## Files
| file | purpose |
|---|---|
| `lib.py` | cohort definition, preprocessing (imputation + scaling fitted on training data), metrics, DeLong, NRI/IDI |
| `dbl.py` | expanding-window helpers (`fit_predict`, `run`, `summ`) |
| `final_numbers.py` | all primary and sensitivity results (Tables 1–3, Online Resource 3) |
| `extra.py`, `approach_recal.py` | missing-data tests/complete-case analysis; approach-specific recalibration and Holm adjustment |
| `make_figs.py` | Figures 1–4 |
| `original_single_split/` | initial single-split pipeline (Online Resource 1) |
| `prepare_data.py` | converts the local Excel database to `data/cohort.pkl` |

## How to run
1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Place the database locally and run `python prepare_data.py database.xlsx` (creates `data/cohort.pkl`; `data/` is git-ignored).
3. Optional: put patient IDs to be excluded, one per line, in `data/exclude_ids.txt` (local file, never committed).
4. `./run_all.sh` (the full run takes several hours because of bootstrap resampling).

### Required columns (Japanese names as in the source database)
`ID`, `age`, `sex` (M/F), `術日` (date of surgery), `術式` (procedure; text used for exclusion rules), `Grade≧2`, `Grade≧3`, `ASA-PS`, `CCI`, `CCI(年齢調整)`, `TG`, `history_of_laparotomy`, `open`, `simultaneous_operation`, `BMI`, `CRP/Alb_ratio`, `PNI`, `PT-INR`.

## Reproducibility
Python 3.13; library versions in `requirements.txt`; fixed random seeds (`SEED = 42`, cross-validation seeds 0–4 for sensitivity). Results may differ slightly across library versions.
