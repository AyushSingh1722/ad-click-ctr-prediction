# Project Handoff — Ad Click CTR Prediction

## Current Status
All 8 notebooks complete, executed, audited, and fixed.
5 Phase 2 READMEs written (docs/04–08).
Main README.md needs Phase 2 update.
Final step: update main README, then push everything to GitHub.

---

## Environment
Path: /mnt/home-ldap/ayush_ldap/projects/laboratory/ad-click-ctr-prediction
Activate: source venv-adclick/bin/activate
Kernel: adclick-venv, Python 3.12.8
GitHub: https://github.com/AyushSingh1722/ad-click-ctr-prediction

---

## How We Work Together
- Work piecewise — one step at a time, confirm before proceeding
- All findings verified against actual cell outputs before writing up
- Every README written in first person
- Forbidden phrases everywhere: "In conclusion", "It is worth noting",
  "Leveraging", "In the realm of", "Delve into", "This notebook aims to",
  "It is important to note", "Utilizing"
- Execution errors go to Claude Chat for diagnosis, fix goes to Claude Code
- Full accuracy audit before any documentation
- Update HANDOFF.md before closing any session

---

## Project Overview
End-to-end ad click CTR prediction across two datasets and two phases.

Phase 1 (COMPLETE): KDD Cup 2012 Track 2 + small advertising dataset
Phase 2 (COMPLETE): Avito Context Ad Clicks 2015 + inference pipeline

Story arc:
- NB01–02: KDD Cup — built CTR from scratch, learned rate features
- NB03: Small dataset — model choice depends on data size
- NB04: Avito EDA — different platform, richer features
- NB05: Avito feature engineering — extending KDD approach
- NB06: Avito modelling — XGBoost, calibration failure discovery
- NB07: Ensemble — Platt scaling + XGBoost + LightGBM
- NB08: Inference pipeline — end-to-end scoring demo

---

## Repo Structure
```
ad-click-ctr-prediction/
├── notebooks/
│   ├── 01_kdd_eda_industrial_scale.ipynb          ✅
│   ├── 02_kdd_feature_engineering_and_modelling.ipynb ✅
│   ├── 03_advertising_eda_and_model_comparison.ipynb  ✅
│   ├── 04_avito_eda.ipynb                         ✅
│   ├── 05_avito_feature_engineering.ipynb         ✅
│   ├── 06_avito_modelling.ipynb                   ✅
│   ├── 07_avito_ensemble.ipynb                    ✅
│   └── 08_avito_inference_pipeline.ipynb          ✅
├── data/
│   ├── track2/
│   ├── sample/ad_click_data.csv
│   └── avito/sample/
│       ├── train_stream_5m.tsv
│       ├── search_info_5m.tsv
│       ├── ads_info_5m.tsv
│       ├── user_info.tsv
│       ├── category.tsv
│       ├── location.tsv
│       ├── features_5m.parquet
│       └── predictions.csv
├── models/
│   ├── xgb_kdd_best_model.joblib
│   ├── xgb_kdd_best_params.json
│   ├── lgbm_advertising_model.joblib
│   ├── xgb_advertising_model.joblib
│   ├── advertising_model_comparison.json
│   ├── xgb_avito_best_model.joblib
│   ├── xgb_avito_model_comparison.json
│   ├── platt_xgb.joblib
│   ├── lgb_avito_model.joblib
│   ├── platt_lgb.joblib
│   └── ensemble_avito_model_comparison.json
├── docs/
│   ├── 01_kdd_eda_readme.md
│   ├── 02_kdd_modelling_readme.md
│   ├── 03_advertising_modelling_readme.md
│   ├── 04_avito_eda_readme.md
│   ├── 05_avito_feature_engineering_readme.md
│   ├── 06_avito_modelling_readme.md
│   ├── 07_avito_ensemble_readme.md
│   └── 08_avito_inference_readme.md
├── requirements.txt
├── .gitignore
├── README.md
└── HANDOFF.md
```

---

## Phase 1 — COMPLETE ✅

### Key Verified Numbers

**NB01 — KDD Cup EDA**
- 4.95M rows after 7-file merge
- Net CTR: 4.2%
- Position: Pos1=5.52%, Pos2=3.08%, Pos3=1.89%, ratio=2.9×
- Depth: D1=4.58%, D2=4.80%, D3=3.56%
- Gender: Male=4.41%, Female=4.59%, Unknown=4.53%
- Age: Group5=4.75%, Group6=5.26% highest; ages 18–30 click LESS than under-18
- Frequent advertisers: median=3.51%, mean=4.14%
- Infrequent advertisers: median=1.58%, mean=3.89%
- CTR peaks at 3-word queries (4.90%)
- 75th pct: query words=4, desc words=25, title words=11

**NB02 — KDD Cup Modelling**
- Best: depth=2, n_estimators=800, lr=0.01
- Test AUC=0.6803, MAPE=7.1841%
- Smoothing: α=0.05, β=75
- 12 p* features (pDepthCTR dropped)
- Feature importance (weight): pUId=856, pQId=655, pTitleId=436,
  pDescId=158, pPosCTR=103, pRPosition=84, pKeyId=76, pAdURL=31, pAdvCTR=1
- Zero importance: pAdCTR, pGender, pAge
- depth=4 AUC=0.6728 (worse than depth=2)

**NB03 — Advertising Dataset**
- 1000 rows, 9 features, 0 missing values
- Train/test: 750/250
- LR=97.2%, LightGBM=97.2%, XGB=94.8%, DT=94.4%
- Key insight: LR ties LightGBM — opposite of KDD Cup result

---

## Phase 2 — COMPLETE ✅

### Key Verified Numbers

**NB04 — Avito EDA**
- Contextual click rate: 0.6142%
- Raw click rate (all rows): 0.2977% (misleading artifact)
- IsContext=0 rows: 2,577,017 — all have zero clicks (artifact)
- IsContext=1 rows: 2,422,983
- HistCTR alone AUC: 0.6640 (one feature vs KDD's 12 for 0.6803)
- Position: Pos1=0.7309%, Pos7=0.4614%, ratio≈1.6× (vs KDD 2.9×)
- Category mismatch CTR (0.3167%) > match (0.2921%) — surprising
- Logged-out (0.3180%) > logged-in (0.2553%) — counterintuitive
- Price: broadly increasing CTR with price
- CRITICAL: Always filter IsContext==1 immediately after loading

**NB05 — Avito Feature Engineering**
- Working rows: 2,422,983
- Output: features_5m.parquet, 21 features + IsClick
- Top correlators with IsClick: ad_ctr=0.0411, category_ctr=0.0397,
  HistCTR=0.0311, location_ctr=0.0298, user_historical_ctr=0.0204
- Global CTR prior: 0.0061
- Zero prior user history: 0.2707 (27.1% of rows)
- session_size: mean=1.87, binary (values 1 and 2 only)
- Price coverage: 99.9%
- location_match: discarded (0.0% match rate, zero variance)
- Replaced with: category_match
- Smoothing: α=0.05, β=75 (same as KDD Cup)

Feature list (exact order — must match in NB06/07/08):
```
FEATURE_COLS = [
    'HistCTR',
    'Position', 'position_in_session', 'ads_before', 'session_size',
    'hour_of_day', 'day_of_week',
    'user_impression_count', 'user_historical_ctr', 'uid_category_count',
    'ad_ctr', 'category_ctr', 'location_ctr', 'position_ctr', 'device_ctr',
    'price_log', 'has_price', 'title_word_count',
    'category_level', 'category_match', 'IsUserLoggedOn'
]
```

**NB06 — Avito Modelling**
- Train: 1,938,386 rows | Test: 484,597 rows (time-based 80/20)
- Baseline log-loss (HistCTR alone): 0.039682
- Best XGBoost log-loss: 0.478329 (12× WORSE than baseline)
- Best XGBoost AUC: 0.758 vs HistCTR AUC 0.664 (+9 points)
- Best params: depth=6, n_estimators=800, lr=0.05 (opposite of KDD depth=2)
- scale_pos_weight: 161.33× (1,926,445 neg / 11,941 pos)
- Top 3 by gain: HistCTR=848.6, user_historical_ctr=735.8, category_ctr=612.9
- position_ctr: zero splits (absent from booster score)
- device_ctr: ranked #18 (gain=240.9)
- Key insight: XGBoost wins on ranking (AUC) but fails on calibration (log-loss)
  because scale_pos_weight=161 pushes predicted probabilities ~55× above true CTR
- Fix: Platt scaling in NB07

**NB07 — Ensemble + Calibration**
- Three-way split: train=1,453,789 / cal=484,597 / test=484,597 (60/20/20)
- XGB raw log-loss: 0.478329 → calibrated: 0.035021 (13.7× reduction)
- LGB raw log-loss: 0.485788 → calibrated: 0.034235 (14.2× reduction)
- Both beat HistCTR baseline (0.039682) ✓
- XGB AUC: 0.7580 (identical before/after Platt — monotone transform)
- LGB AUC: 0.7507 (identical before/after Platt)
- Ensemble log-loss: 0.034040 (beats both individual models)
- Ensemble AUC: 0.7613 (beats both individual models)
- Both success criteria met: log-loss < 0.039682 ✓, AUC > 0.758 ✓
- Raw proba overinflation: ~55× true CTR (XGB mean=0.3329 vs CTR=0.006071)
- LGB top 3 by gain: HistCTR (14,757,060), ad_ctr (4,052,276), category_ctr (4,042,614)
- Key divergence: user_historical_ctr XGB #2 vs LGB #10
- Note: XGBoost Platt calibration is impure (cal fold was inside NB06 training data)
  LightGBM calibration is fully clean

**NB08 — Inference Pipeline**
- Single record ensemble: 0.7565% (+23% above 0.6142% baseline)
- XGB raw: 0.521395 (~87× inflated) → calibrated: 0.006775
- LGB raw: 0.517250 → calibrated: 0.008355
- Position gradient: INVERTED — model Pos1/Pos7=0.80× vs empirical 1.58×
  Confound: higher positions co-occur with larger session_size
- HistCTR sensitivity: 1.33× range, non-monotone across 20× HistCTR span
- Batch calibration ratio: 1.015× (mean predicted 0.6089% vs actual 0.6000%)
- Predictions saved: data/avito/sample/predictions.csv

---

## Saved Models
- models/xgb_kdd_best_model.joblib
- models/xgb_kdd_best_params.json
- models/lgbm_advertising_model.joblib
- models/xgb_advertising_model.joblib
- models/advertising_model_comparison.json
- models/xgb_avito_best_model.joblib
- models/xgb_avito_model_comparison.json
- models/platt_xgb.joblib
- models/lgb_avito_model.joblib
- models/platt_lgb.joblib
- models/ensemble_avito_model_comparison.json

---

## Documentation Status
- docs/01_kdd_eda_readme.md ✅
- docs/02_kdd_modelling_readme.md ✅
- docs/03_advertising_modelling_readme.md ✅
- docs/04_avito_eda_readme.md ✅
- docs/05_avito_feature_engineering_readme.md ✅
- docs/06_avito_modelling_readme.md ✅
- docs/07_avito_ensemble_readme.md ✅
- docs/08_avito_inference_readme.md ✅
- README.md — needs Phase 2 update (next step)

---

## What To Do Next

### Step 1: Update main README.md
Add Phase 2 to the "The Journey" section.
Add 3 key Phase 2 findings to Key Findings.
Keep it high-level — details live in docs/.

Key findings to add:
1. HistCTR alone (AUC=0.6640) captured 97% of the signal achievable 
   with 12 carefully engineered features (AUC=0.758) — one pre-computed 
   feature does most of the work
2. XGBoost won on ranking (AUC 0.664→0.758) but failed on calibration 
   (log-loss 12× worse than baseline) due to scale_pos_weight distortion;
   Platt scaling fixed it (13.7× log-loss reduction)
3. Ensemble of calibrated XGBoost + LightGBM beat both individual models 
   on both metrics: log-loss=0.034040, AUC=0.7613

### Step 2: Push everything to GitHub
```bash
cd /mnt/home-ldap/ayush_ldap/projects/laboratory/ad-click-ctr-prediction
git add docs/04_avito_eda_readme.md \
        docs/05_avito_feature_engineering_readme.md \
        docs/06_avito_modelling_readme.md \
        docs/07_avito_ensemble_readme.md \
        docs/08_avito_inference_readme.md \
        README.md \
        HANDOFF.md
git commit -m "Add Phase 2 docs, READMEs NB04-08, update main README"
git push origin main
```

---

## Execution Command Template
```bash
cd /mnt/home-ldap/ayush_ldap/projects/laboratory/ad-click-ctr-prediction && \
source venv-adclick/bin/activate && \
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=adclick-venv \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/[notebook].ipynb
```