# Notebook 08 — Avito CTR Prediction: Inference Pipeline

## 1. Why This Notebook Exists

NB07 produced calibrated models that beat the HistCTR baseline on both log-loss and AUC. But offline metrics on a held-out test fold only tell part of the story. Before this project can be called production-ready, I needed to answer three questions: Can a single unseen ad record be scored end-to-end in real time? Do the model's predictions behave sensibly when input features are varied? And does the batch output land near the true click rate rather than diverging from it?

---

## 2. Dataset

`data/avito/sample/features_5m.parquet` — NB05 output, used here for batch scoring only.
Models loaded from disk:
- `models/xgb_avito_best_model.joblib`
- `models/platt_xgb.joblib`
- `models/lgb_avito_model.joblib`
- `models/platt_lgb.joblib`

---

## 3. Approach

I built two components:

1. **`engineer_features(record, global_ctr, alpha, beta)`** — replicates the NB05 feature engineering logic for a single raw ad record. Takes a dictionary with raw fields (HistCTR, Position, SearchDate, IsUserLoggedOn, price, title word count, etc.) and returns a one-row DataFrame with all 21 features in the exact column order expected by the models.

2. **`score_record(record)`** — calls `engineer_features`, runs both XGBoost and LightGBM, applies both Platt scalers, and returns a structured result: xgb_raw, lgb_raw, xgb_cal, lgb_cal, ensemble (simple average), and p_click_pct (ensemble expressed as a percentage).

I then ran three sensitivity analyses — varying Position, HistCTR, and IsUserLoggedOn — to verify that the pipeline's directional predictions are coherent with the patterns identified in NB04 EDA. Finally, I scored the full 484,597-row test fold and measured the batch calibration ratio.

---

## 4. Key Findings

**Single record scoring:**
- Ensemble prediction: **0.7565%** — +23% above the dataset baseline of 0.6142%
- XGBoost raw: **0.521395** (~87× above true CTR of 0.006142) → calibrated: **0.006775**
- LightGBM raw: **0.517250** → calibrated: **0.008355**

**Position gradient (sweep Pos1–Pos7, all other features fixed):**
- Model output: Pos1=**0.7565%**, Pos7=**0.9460%**, ratio=**0.80×** — INVERTED
- Empirical gradient (NB04): Pos1=0.7309%, Pos7=0.4614%, ratio=**1.58×**
- The model predicts that position 7 ads are more likely to be clicked than position 1 ads — the opposite of the observed data pattern. The most likely cause: higher position values (6, 7) co-occur with larger session_size values in the training data. Larger sessions have more engaged users, and the model learned this confound rather than the pure position penalty.

**HistCTR sensitivity (sweep 0.001–0.020, 10 points):**
- Range: approximately **1.33×** from lowest to highest predicted CTR
- The relationship is non-monotone: there is a dip at HistCTR=0.006 and 0.010 before the prediction rises at higher values
- Across a 20× span of HistCTR input, the ensemble output shifts by only 1.33× — HistCTR's marginal contribution is modest once other features are held fixed

**Login effect:**
- Logged-out: **0.9118%** vs Logged-in: **0.7565%**
- Logged-out users are predicted more likely to click — consistent with the counterintuitive finding from NB04 (logged-out CTR=0.3180% vs logged-in CTR=0.2553%)

**Batch calibration:**
- Mean predicted CTR: **0.6089%** vs mean actual CTR: **0.6000%** — ratio **1.015×**
- Near-perfect calibration: the ensemble overshoots by 1.5% relative on average

**Output saved:** `data/avito/sample/predictions.csv`

---

## 5. Challenges and How I Addressed Them

**Feature column ordering.** XGBoost and LightGBM store feature names from training. When `predict_proba` receives a DataFrame, both frameworks match by column name rather than column position. Passing a numpy array without names would use positional matching and silently produce wrong predictions if column order drifted between training and inference. `engineer_features` returns a DataFrame with explicit column names, eliminating this risk.

**Replicating NB05 logic without data.** Inference operates on a single record, not a full dataset. The cumulative user behaviour features (user_impression_count, user_click_count, user_historical_ctr, uid_category_count) are computed from history in NB05 but must be supplied as inputs at inference time — there is no historical data to aggregate over. `engineer_features` accepts these as caller-provided inputs and applies the same global CTR fallback (0.0061) when they are absent. This is correct for a stateless inference endpoint but means the function's user-behaviour features are only as good as the upstream system that maintains per-user running statistics.

**Position inversion.** The position gradient anomaly required investigation rather than dismissal. I verified the direction by running the sweep twice and confirmed the training data has a genuine session_size correlation with higher positions (later-position ads appear in sessions that have already generated multiple impressions, which selects for more active users). This is a training data confound, not a pipeline bug — the model learned a real pattern, but it is not the pattern that would generalise to deployment.

---

## 6. Techniques Used

- Stateless inference function with explicit feature engineering replication
- DataFrame-based `predict_proba` calls to preserve feature-name alignment
- Sensitivity analysis by single-feature sweep with all other features held fixed
- Batch scoring with mean calibration ratio check
- `joblib.load` for model deserialisation
- `numpy.log1p` for price transformation at inference time (mirrors NB05)

---

## 7. What I Would Do Differently

**Add schema validation at the inference boundary.** `engineer_features` currently trusts that the input dictionary has the expected keys. A production function would validate the schema on entry — required fields present, types correct, values in expected ranges — and raise a descriptive error rather than propagating a KeyError or silent NaN through the feature matrix.

**Maintain rate features as a separate service.** The rate-encoded CTRs (ad_ctr, category_ctr, location_ctr, device_ctr) are computed at NB05 training time and frozen. In production these would need to be updated periodically as new impression data arrives; a stale ad_ctr for a new ad would fall back to the Laplace-smoothed prior, which is correct but suboptimal. The right design is a feature store that serves fresh rate estimates at inference time.

**Schedule recalibration.** Platt scaling was fit on data from 2015. Click rates shift over time — seasonality, platform changes, advertiser mix. The calibrated probabilities will drift from reality as the data distribution changes. A production system would monitor the calibration ratio (predicted mean CTR vs actual mean CTR) and trigger recalibration when it drifts beyond a threshold.

**Separate model versioning from inference code.** The inference pipeline hardcodes model paths. A production system would load models by version tag from a model registry, allowing rollback without code changes and supporting A/B testing between model versions.
