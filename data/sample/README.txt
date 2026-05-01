KDD Cup 2012 Track 2 — Data Setup Instructions
===============================================

The notebooks in this repo use two separate datasets. Neither is included
in the repository. Download instructions are below.

──────────────────────────────────────────────
DATASET 1 — KDD Cup 2012 Track 2 (industrial scale)
Used by: notebooks/01_kdd_eda_industrial_scale.ipynb
         notebooks/02_kdd_feature_engineering_and_modelling.ipynb
──────────────────────────────────────────────

Source:
  https://www.kaggle.com/c/kddcup2012-track2

After downloading, place the files in the following layout under data/:

  data/
  ├── track2/
  │   └── track2/
  │       ├── training.txt                      (main training log, ~2.5 GB)
  │       ├── test.txt                          (test instances)
  │       ├── userid_profile.txt                (user gender & age)
  │       ├── queryid_tokensid.txt              (search query tokens)
  │       ├── titleid_tokensid.txt              (ad title tokens)
  │       ├── descriptionid_tokensid.txt        (ad description tokens)
  │       └── purchasedkeywordid_tokensid.txt   (purchased keyword tokens)
  └── KDD_Track2_solution.csv                   (released post-competition)

The notebooks cap the row read to 5,000,000 training rows and 1,000,000
test rows, so you do not need to load the full files.

──────────────────────────────────────────────
DATASET 2 — Advertising Click Dataset (small, 1,000 rows)
Used by: notebooks/03_advertising_eda_and_model_comparison.ipynb
──────────────────────────────────────────────

Source (two equivalent copies exist on Kaggle):
  https://www.kaggle.com/farhanmd29/predicting-customer-ad-clicks
  https://www.kaggle.com/imprime/logistic-regression-with-ad-click-dataset

After downloading, place the file here:

  data/
  └── Ad Click Data.csv

The notebook currently has a hardcoded Kaggle path (../input/...) which
will be fixed in the next cleanup phase to point to data/Ad Click Data.csv.

──────────────────────────────────────────────
NOTE: data/ is listed in .gitignore (or should be) — never commit raw data.
