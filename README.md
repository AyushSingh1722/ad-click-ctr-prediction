# Ad Click CTR Prediction

Two datasets, three notebooks, one question: what actually predicts whether someone clicks an ad. I worked through the KDD Cup 2012 Track 2 dataset — five million anonymised search ad impressions — building smoothed historical CTR features and training a gradient-boosted model to AUC 0.6803. A follow-up on a clean 1,000-row advertising dataset found the model hierarchy reversed completely, which is the more interesting result.

---

## Project Structure

```
ad-click-ctr-prediction/
├── notebooks/
│   ├── 01_kdd_eda_industrial_scale.ipynb               # EDA on 4.95M merged KDD rows
│   ├── 02_kdd_feature_engineering_and_modelling.ipynb  # p* feature engineering + XGBoost
│   └── 03_advertising_eda_and_model_comparison.ipynb   # Four-model comparison on small dataset
├── data/
│   ├── track2/                                         # KDD Cup raw files (not in repo — too large)
│   └── sample/
│       └── ad_click_data.csv                           # Small advertising dataset (1,000 rows)
├── models/
│   ├── xgb_kdd_best_model.joblib                       # Trained KDD XGBoost model
│   ├── xgb_kdd_best_params.json                        # Grid search best hyperparameters
│   ├── lgbm_advertising_model.joblib                   # LightGBM model for advertising dataset
│   ├── xgb_advertising_model.joblib                    # XGBoost model for advertising dataset
│   └── advertising_model_comparison.json               # Accuracy comparison across four models
├── docs/                                               # Per-notebook write-ups
├── requirements.txt
└── .gitignore
```

Per-notebook write-ups are in [`docs/`](docs/), covering approach, findings, and what I would do differently for each notebook.

---

## Key Findings

- At industrial scale, who the user is and what they searched for matters more than where the ad appears — pUId, pQId, and pTitleId are the three dominant features while gender and age contribute zero model importance despite being available
- Converting anonymised entity IDs to smoothed historical CTR features is what makes the KDD Cup data learnable; without that step there is nothing to train on
- On clean, low-dimensional data logistic regression matched the best gradient boosted model exactly — the model hierarchy is a function of the data, not a fixed ranking

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

Run the notebooks in order: 01 → 02 → 03. Notebook 02 depends on the merged dataframe produced in 01, and 03 is self-contained.

---

## Acknowledgements

Some ideas around feature engineering — particularly the smoothed historical CTR approach and the `p*` feature naming convention — were inspired by open-source Kaggle competition work on large-scale CTR prediction problems. The smoothing formulation draws on techniques that became standard practice in the field following competitions like KDD Cup 2012 and related industry work on ad click modelling.
