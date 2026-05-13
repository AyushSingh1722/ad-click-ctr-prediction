# Ad Click CTR Prediction

Three datasets, eight notebooks, one question: what actually predicts whether someone clicks an ad. Phase 1 worked through the KDD Cup 2012 Track 2 dataset — five million anonymised search ad impressions — building smoothed historical CTR features and training a gradient-boosted model to AUC 0.6803. A follow-up on a clean 1,000-row advertising dataset found the model hierarchy reversed completely. Phase 2 applied the same pipeline to the Avito Context Ad Clicks 2015 dataset, added user behaviour features and content signals, discovered a calibration failure in the baseline model, and fixed it with an ensemble of Platt-scaled XGBoost and LightGBM — then wired the whole thing into an end-to-end inference pipeline.

---

## Project Structure

```
ad-click-ctr-prediction/
├── notebooks/
│   ├── 01_kdd_eda_industrial_scale.ipynb               # EDA on 4.95M merged KDD rows
│   ├── 02_kdd_feature_engineering_and_modelling.ipynb  # p* feature engineering + XGBoost
│   ├── 03_advertising_eda_and_model_comparison.ipynb   # Four-model comparison on small dataset
│   ├── 04_avito_eda.ipynb                              # Avito EDA — IsContext artifact, HistCTR baseline
│   ├── 05_avito_feature_engineering.ipynb              # 21 features across 5 families, 2.4M rows
│   ├── 06_avito_modelling.ipynb                        # XGBoost grid search + calibration failure discovery
│   ├── 07_avito_ensemble.ipynb                         # Platt scaling + LightGBM + ensemble
│   └── 08_avito_inference_pipeline.ipynb               # End-to-end single-record and batch scoring
├── data/
│   ├── track2/                                         # KDD Cup raw files (not in repo — too large)
│   ├── sample/
│   │   └── ad_click_data.csv                           # Small advertising dataset (1,000 rows)
│   └── avito/sample/                                   # Avito raw files (not in repo — too large)
│       ├── train_stream_5m.tsv
│       ├── search_info_5m.tsv
│       ├── ads_info_5m.tsv
│       ├── user_info.tsv
│       ├── category.tsv
│       ├── location.tsv
│       ├── features_5m.parquet                         # Output of NB05, input to NB06–08
│       └── predictions.csv                             # Output of NB08 batch scoring
├── models/
│   ├── xgb_kdd_best_model.joblib                       # Trained KDD XGBoost model
│   ├── xgb_kdd_best_params.json                        # Grid search best hyperparameters
│   ├── lgbm_advertising_model.joblib                   # LightGBM model for advertising dataset
│   ├── xgb_advertising_model.joblib                    # XGBoost model for advertising dataset
│   ├── advertising_model_comparison.json               # Accuracy comparison across four models
│   ├── xgb_avito_best_model.joblib                     # Avito XGBoost (depth=6, n_est=800)
│   ├── xgb_avito_model_comparison.json                 # Grid search results across 6 configs
│   ├── platt_xgb.joblib                                # Platt scaler for XGBoost
│   ├── lgb_avito_model.joblib                          # Avito LightGBM model
│   ├── platt_lgb.joblib                                # Platt scaler for LightGBM
│   └── ensemble_avito_model_comparison.json            # Raw vs calibrated vs ensemble metrics
├── docs/                                               # Per-notebook write-ups (NB01–08)
├── requirements.txt
└── .gitignore
```

Per-notebook write-ups are in [`docs/`](docs/), covering approach, findings, and what I would do differently for each notebook.

---

## The Journey

**Phase 1 — KDD Cup 2012 + Advertising Dataset**

NB01 merged seven KDD Cup files into 4.95M rows and established that user identity (pUId) and query (pQId) dominate position and demographics. NB02 converted anonymised IDs into Laplace-smoothed historical CTR features — the step that makes the data learnable — and found that depth=2 XGBoost outperforms deeper trees on an almost-entirely-smoothed feature set (AUC=0.6803). NB03 tested four models on a clean 1,000-row advertising dataset and found logistic regression tied LightGBM, reversing the hierarchy entirely.

**Phase 2 — Avito Context Ad Clicks 2015**

NB04 discovered the IsContext artifact: 2.577M of the 5M rows have structural zero clicks and must be filtered before any analysis. After filtering, HistCTR alone reached AUC=0.6640 — a single pre-computed feature matching most of what twelve engineered features achieved on KDD Cup. NB05 engineered 21 features across five families (baseline, temporal, user behaviour, rate encoding, content) with leakage-safe cumulative statistics. NB06 ran a depth × n_estimators grid search, found depth=6 beats KDD Cup's depth=2 on the richer Avito feature set, and exposed the calibration failure: XGBoost improved AUC by 9 points but made log-loss 12× worse. NB07 fixed it — Platt scaling on a clean 60/20/20 split reduced log-loss by 13.7× and the ensemble beat both individual models on both metrics. NB08 wired the pipeline into an end-to-end scorer, ran sensitivity analyses on Position, HistCTR, and login status, and confirmed near-perfect batch calibration (ratio=1.015×).

---

## Key Findings

**Phase 1 — KDD Cup / Advertising**

- At industrial scale, who the user is and what they searched for matters more than where the ad appears — pUId, pQId, and pTitleId are the three dominant features while gender and age contribute zero model importance despite being available
- Converting anonymised entity IDs to smoothed historical CTR features is what makes the KDD Cup data learnable; without that step there is nothing to train on
- On clean, low-dimensional data logistic regression matched the best gradient boosted model exactly — the model hierarchy is a function of the data, not a fixed ranking

**Phase 2 — Avito**

- HistCTR alone (AUC=0.6640) captured 97% of the signal achievable with 12 carefully engineered features (AUC=0.758) — one pre-computed feature does most of the work on Avito
- XGBoost won on ranking (AUC 0.664→0.758) but failed on log-loss (12× worse than baseline) due to scale_pos_weight distortion; Platt scaling reduced log-loss by 13.7×
- Ensemble of calibrated XGBoost + LightGBM beat both individual models on both metrics: log-loss=0.034040, AUC=0.7613

---

## How to Run

**Requirements:** Python 3.12, Jupyter, and the packages in `requirements.txt`.

```bash
# Clone the repo
git clone https://github.com/AyushSingh1722/ad-click-ctr-prediction.git
cd ad-click-ctr-prediction

# Create and activate the virtual environment
python3 -m venv venv-adclick
source venv-adclick/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register the kernel
python -m ipykernel install --user --name=adclick-venv --display-name "adclick-venv"

# Launch Jupyter
jupyter notebook
```

**KDD Cup data:** The Track 2 files are not in this repository (too large for GitHub). Download them from [Kaggle KDD Cup 2012 Track 2](https://www.kaggle.com/c/kddcup2012-track2) and place them under `data/track2/`. The notebook expects the raw files there and will load them directly.

**Notebook 03** runs entirely on `data/sample/ad_click_data.csv`, which is included in the repo. No additional downloads needed.

**Avito data:** The Avito raw files are not in this repository. Download them from [Kaggle Avito Context Ad Clicks](https://www.kaggle.com/c/avito-context-ad-clicks) and place the six TSV files under `data/avito/sample/`. NB04 expects them there and will merge them on load.

Run the notebooks in order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08. Each Phase 2 notebook depends on outputs from the previous one: NB05 reads the merged dataframe from NB04, NB06–08 read `features_5m.parquet` written by NB05, and NB07–08 load the models saved by NB06.

---

## Running the API

The FastAPI service loads the trained ensemble and exposes three endpoints.

```bash
# Install API dependencies
pip install -r requirements_api.txt

# Start the server (from repo root, venv active)
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Confirms models loaded, returns ensemble AUC and log-loss |
| `/predict` | POST | Scores a single ad impression → calibrated P(click) |
| `/batch_predict` | POST | Scores up to 10,000 records → predictions list |

Interactive docs (Swagger UI) at `http://localhost:8000/docs` — every endpoint testable with a form, no curl needed.

Sample request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "HistCTR": 0.008,
    "Position": 1,
    "IsUserLoggedOn": 1,
    "Price": 4500,
    "Title": "Продам ноутбук Lenovo ThinkPad",
    "category_level": 2,
    "category_match": 1,
    "session_size": 2,
    "ad_ctr": 0.009,
    "category_ctr": 0.007,
    "location_ctr": 0.006,
    "SearchDate": "2015-06-01 14:00:00"
  }'
```

---

## Running the Demo

The Streamlit app scores ads interactively using the same ensemble. Adjust any input and all three panels update live — no page reload.

```bash
# From repo root, venv active
streamlit run src/app.py --server.port 8003
# Open http://localhost:8003
```

Three panels:
- **Prediction** — calibrated ensemble CTR vs 0.6142% baseline, with XGBoost and LightGBM breakdown
- **Position Sensitivity** — sweep Position 1→7 and watch the inverted gradient documented in NB08 appear live
- **HistCTR Sensitivity** — sweep HistCTR across 5 values and observe the non-monotone response documented in NB08

The app calls `engineer_features()` and the 4 saved models directly — no dependency on the API being running.

---

## Acknowledgements

Some ideas around feature engineering — particularly the smoothed historical CTR approach and the `p*` feature naming convention — were inspired by open-source Kaggle competition work on large-scale CTR prediction problems. The smoothing formulation draws on techniques that became standard practice in the field following competitions like KDD Cup 2012 and related industry work on ad click modelling.
