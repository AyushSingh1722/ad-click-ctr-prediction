# Notebook 06 — Avito CTR Prediction: Modelling

## 1. Why This Notebook Exists

NB05 produced a 21-feature matrix. The question was whether those features actually improve over HistCTR alone, and whether the KDD Cup finding — that shallow trees (depth=2) beat deeper ones — generalises to Avito's richer feature set. I also needed to discover where the calibration problem lives before NB07 could fix it.

---

## 2. Dataset

`data/avito/sample/features_5m.parquet` — output of NB05.
Time-based 80/20 split (chronological, not random — user behaviour features are causal):

| Split | Rows |
|-------|------|
| Train | 1,938,386 |
| Test  | 484,597 |

---

## 3. Approach

I established the HistCTR-alone log-loss as the baseline before training anything, then ran a 6-configuration grid search over `max_depth ∈ {2, 4, 6}` and `n_estimators ∈ {500, 800}`. Learning rate was fixed at 0.05 for runtime reasons. `scale_pos_weight = neg/pos = 161.33` was applied to correct for the severe 161:1 class imbalance.

After identifying the best configuration, I extracted feature importances by gain and reported AUC alongside log-loss to separate the ranking question (does XGBoost discriminate better than HistCTR alone?) from the calibration question (does it produce better probabilities?).

---

## 4. Key Findings

**Baseline:**
- HistCTR-alone log-loss: **0.039682** — deceptively low because HistCTR is already a near-perfectly calibrated empirical probability

**Best XGBoost configuration:**
- Params: **depth=6, n_estimators=800** — the opposite of KDD Cup's winning depth=2
- Log-loss: **0.478329** — **12× worse** than the HistCTR baseline
- AUC: **0.7580** — **+9.4 points** above the HistCTR-alone baseline of 0.664
- scale_pos_weight: **161.33×** (1,926,445 negatives / 11,941 positives)

**The calibration finding:** XGBoost wins decisively on ranking (AUC) but fails on calibration (log-loss). `scale_pos_weight=161` upweights positive-class gradients during training, pushing predicted probabilities far above the true CTR of 0.6142%. Log-loss directly measures calibration quality; a model predicting 33% click probability where the true rate is 0.6% is catastrophically penalised.

**Feature importance by gain (top 3):**
1. HistCTR: **848.6**
2. user_historical_ctr: **735.8** — a user behaviour feature, not a rate-encoding feature
3. category_ctr: **612.9**

**Notable absences and low-rank features:**
- position_ctr: **zero splits** — absent from the booster score entirely
- device_ctr: **#18**, gain=**240.9**

**Depth finding:**
- Log-loss improved monotonically with depth: depth=2 (0.577) → depth=4 (0.543) → depth=6 (0.478)
- Avito's richer feature set (real prices, ad titles, device info) provides enough genuine signal for deeper splits; KDD Cup's almost-entirely-smoothed-CTR feature set did not

---

## 5. Challenges and How I Addressed Them

**The log-loss paradox.** HistCTR is not just a feature — it is already a calibrated probability. Its log-loss floor of 0.039682 is extremely low because it reflects empirical click rates, not model-distorted scores. Any model that uses `scale_pos_weight` to handle class imbalance will inflate its probability outputs and therefore degrade log-loss, even while improving AUC. I documented this explicitly and deferred the calibration fix to NB07 (Platt scaling) rather than trying to solve ranking and calibration in one step.

**Time-based split requirement.** The user behaviour features (user_historical_ctr, user_impression_count, uid_category_count) are running cumulative statistics computed chronologically in NB05. A random split would allow later rows — which have higher cumulative counts that encode future behaviour — to appear in the training set, leaking information from the future into model training. The 80/20 split preserves chronological ordering throughout.

**position_ctr zero importance.** position_ctr was engineered in NB05 but contributed zero splits to the best XGBoost model. This is consistent with `Position` itself being in the feature set — the model may extract the position signal directly from the raw position value rather than from the smoothed CTR version.

---

## 6. Techniques Used

- XGBoost with `tree_method='hist'` for speed on 2.4M rows
- `scale_pos_weight = neg/pos` for class imbalance
- Grid search over depth × n_estimators (6 configurations)
- `get_booster().get_score(importance_type='gain')` for feature importance
- `sklearn.metrics.log_loss` and `roc_auc_score` for dual-metric evaluation
- Time-based split to respect causal ordering

---

## 7. What I Would Do Differently

**Fix calibration in the same notebook.** Separating calibration into NB07 made the story cleaner but created a practical problem: the XGBoost model in NB06 was trained on 80% of the data, which later became the calibration fold for Platt scaling in NB07. A cleaner design would have trained XGBoost on 60% from the start, reserving 20% for calibration and 20% for test — the same three-way split NB07 ultimately uses for LightGBM.

**Add early stopping.** The grid searched a fixed n_estimators rather than using early stopping with a validation fold. Early stopping would find the true optimal tree count automatically and avoid the risk of overfitting at high n_estimators values — especially relevant at depth=6.

**Investigate position_ctr's zero importance.** The feature took significant engineering effort but contributed nothing. Understanding why (likely redundancy with raw Position) would inform whether to keep it in future feature sets or drop it earlier in the pipeline.
