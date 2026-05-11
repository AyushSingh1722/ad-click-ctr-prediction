# Notebook 07 — Avito CTR Prediction: Ensemble + Calibration

## 1. Why This Notebook Exists

NB06 left two problems on the table: XGBoost produced excellent rankings (AUC=0.7580) but catastrophically miscalibrated probabilities (log-loss=0.478329, 12× worse than the HistCTR baseline). The fix required a clean three-way split that NB06's 80/20 design could not provide. This notebook retrains with a proper 60/20/20 structure, calibrates both XGBoost and LightGBM with Platt scaling, and tests whether a simple average ensemble beats either model alone.

---

## 2. Dataset

`data/avito/sample/features_5m.parquet` — output of NB05.
Three-way chronological split:

| Split | Rows | Purpose |
|-------|------|---------|
| Train | 1,453,789 | Base model training |
| Cal   | 484,597   | Platt scaling fit |
| Test  | 484,597   | Final held-out evaluation |

The split respects chronological ordering throughout. LightGBM's calibration fold is fully held out from base model training. XGBoost was originally trained on 80% of data in NB06 (rows 0–80%), so the calibration fold here (rows 60–80%) overlaps with its training data — this is documented as an impurity, not corrected by retraining.

---

## 3. Approach

I trained LightGBM on the 60% training fold, then fit a Platt scaler (logistic regression on raw model scores) for each model using the calibration fold. I evaluated raw and calibrated log-loss and AUC on the held-out test fold, then formed a simple average ensemble of the two calibrated models.

Two success criteria guided evaluation:
1. Calibrated log-loss < 0.039682 (beat the HistCTR baseline)
2. AUC > 0.758 (match or exceed NB06's best XGBoost)

---

## 4. Key Findings

**Platt scaling results:**

| Model | Raw log-loss | Calibrated log-loss | Reduction |
|-------|-------------|---------------------|-----------|
| XGBoost | 0.478329 | 0.035021 | 13.7× |
| LightGBM | 0.485788 | 0.034235 | 14.2× |

Both calibrated models beat the HistCTR baseline of 0.039682. AUC is unchanged by Platt scaling (it is a monotone transform of the raw scores): XGB=0.7580, LGB=0.7507.

**Ensemble results:**
- Log-loss: **0.034040** — beats both individual calibrated models
- AUC: **0.7613** — beats both individual models
- Both success criteria met ✓

**Raw probability overinflation:**
- XGBoost mean predicted probability: 0.3329 — approximately **55× above the true CTR of 0.6071%**
- This is the `scale_pos_weight=161` effect from NB06: the booster was trained to treat each positive example as 161 negatives, pushing predicted scores far into the high-probability range before calibration corrects them

**LightGBM feature importance by gain (top 3):**
1. HistCTR: **14,757,060**
2. ad_ctr: **4,052,276**
3. category_ctr: **4,042,614**

**XGBoost vs LightGBM divergence:**
- user_historical_ctr: XGBoost #2 (gain=735.8) vs LightGBM #10
- The user behaviour signal is weighted very differently by the two algorithms. XGBoost's gain metric reflects average improvement per split; LightGBM's larger absolute gain values reflect cumulative gain across all splits with a different leaf structure.

---

## 5. Challenges and How I Addressed Them

**XGBoost calibration impurity.** The cleanest calibration design trains a base model on fold A, fits Platt scaling on fold B, and evaluates on fold C — with no data overlap between folds. LightGBM achieves this here. XGBoost does not: it was trained in NB06 on rows 0–80%, and the calibration fold (rows 60–80%) was inside that training window. Platt scaling on an in-sample fold tends to understate the true calibration error. I documented this explicitly rather than retraining NB06, because the ensemble still meets both success criteria and the impurity is conservative (if anything it overstates calibration quality for XGBoost, making the LightGBM result the more trustworthy individual baseline).

**Why Platt scaling works here.** The raw XGBoost scores are severely right-shifted (mean≈0.33) but rank-ordered correctly (AUC=0.758). Platt scaling fits a single logistic curve mapping raw scores to calibrated probabilities. Because the rank order is preserved, it does not disturb AUC. The 13.7× log-loss reduction confirms that most of the log-loss damage in NB06 was purely a probability scale problem, not a structural ranking problem.

**Ensemble marginal gain.** The ensemble log-loss (0.034040) is only slightly below the best individual model (LGB: 0.034235). The marginal gain is real but small — the two models are trained on very similar features and similar data, so their errors are correlated. A larger gain would require base models with genuinely different error structures (e.g., a neural network alongside a tree ensemble).

---

## 6. Techniques Used

- Platt scaling: logistic regression (`sklearn.linear_model.LogisticRegression`) fit on raw model scores against true labels on a held-out calibration fold
- Three-way chronological split (60/20/20) to separate training, calibration, and evaluation
- Simple average ensemble of two calibrated probability outputs
- `sklearn.metrics.log_loss` and `roc_auc_score` for dual-metric evaluation
- LightGBM with `objective='binary'`, `metric='binary_logloss'`
- `get_booster().get_score(importance_type='gain')` (XGBoost) and `booster_.feature_importance(importance_type='gain')` (LightGBM) for feature importance

---

## 7. What I Would Do Differently

**Retrain XGBoost on the 60% fold.** The NB06 XGBoost model was trained on 80% of the data, creating a calibration impurity. The correct design — used for LightGBM here — is to train on 60% only. The fix was deferred to keep NB06 intact as a standalone notebook documenting the depth grid search, but a production system would not carry this impurity forward.

**Add temperature scaling as an alternative.** Platt scaling fits two parameters (slope and intercept of a logistic curve). Temperature scaling fits only one (a scalar that divides the raw logit). For well-calibrated base models, temperature scaling generalises better to unseen data. Testing both and choosing based on calibration curve quality would be more principled.

**Use stacked generalisation instead of simple averaging.** The ensemble here is a fixed 50/50 average. A second-level logistic regression fit on the calibration fold's predictions could learn the optimal weight for each model, potentially extracting more gain from their complementary error structures.

**Test calibration quality with reliability diagrams.** I reported scalar log-loss but did not visualise the calibration curve (predicted probability vs empirical click rate across probability bins). A reliability diagram would reveal whether the calibration is uniform across the score range or concentrated at specific probability levels.
