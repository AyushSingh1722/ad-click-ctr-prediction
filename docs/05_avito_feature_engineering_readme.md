# Notebook 05 — Avito CTR Prediction: Feature Engineering

## 1. Why This Notebook Exists

NB04 showed that HistCTR alone achieves AUC=0.6640. That is a strong single-feature baseline, but it says nothing about calibration — and Avito's competition metric is log-loss, not AUC. I needed to build features that either (a) improve ranking beyond what HistCTR alone provides, or (b) add information the model can use to calibrate its probability outputs toward the true 0.6142% CTR.

The engineering approach is grounded in three sources: the rate-encoding pipeline from KDD Cup NB02 (smoothed entity CTRs), the user behaviour features from the Gzsiceberg reference pipeline (chronological running totals), and signals unique to Avito that did not exist in KDD Cup (price, ad title, category hierarchy).

---

## 2. Dataset

Same 6-file merge as NB04. Working set: **2,422,983 rows** (IsContext=1 only).

Output: `data/avito/sample/features_5m.parquet` — 21 features + IsClick target + 2 ID columns.

---

## 3. Approach

I organised features into five families and engineered each in sequence before assembling the final matrix:

1. **Baseline**: HistCTR (pass-through) and IsUserLoggedOn
2. **Temporal/session**: hour of day, day of week, session size, position in session, ads seen before current position
3. **User behaviour**: cumulative impression count, click count, historical CTR, and per-category impression count — all computed with a chronological sort and `shift(1)` to prevent leakage
4. **Rate encoding**: Laplace-smoothed CTRs for ad, category, location, position, and device — same α=0.05, β=75 as KDD Cup NB02
5. **Content**: price (log-transformed), has-price flag, ad title word count, category level, and category match

Each family was validated on correlation with IsClick before inclusion. The final matrix was written to parquet and verified for NaN coverage and schema before handing off to NB06.

---

## 4. Key Findings

**Working set:**
- Rows used for feature engineering: **2,422,983**

**Feature correlations with IsClick (Pearson |r|):**
- ad_ctr: **0.0411** (highest)
- category_ctr: **0.0397**
- HistCTR: **0.0311**
- location_ctr: **0.0298**
- user_historical_ctr: **0.0204**

These are all small in absolute terms — typical for CTR datasets where the signal-to-noise ratio is very low. Rate-encoded features dominate the top of the correlation ranking.

**User behaviour coverage:**
- Global CTR prior used as fallback: **0.0061**
- Rows with zero prior user history (fallback to prior): **27.1%** of contextual impressions (0.2707 fraction)
- This means the user_historical_ctr feature carries real information for 72.9% of rows and falls back to the global prior for the rest

**Session features:**
- session_size mean: **1.87**; binary — only values 1 and 2 appear in this dataset
- This is more constrained than expected and means session_size provides limited granularity

**Content coverage:**
- Price coverage: **99.9%** of contextual ads have a listed price — nearly universal, so has_price is a near-constant flag
- category_match rate: **75.9%** of contextual impressions (1,839,179 rows)

**Discarded feature:**
- location_match: **0.0% match rate**, zero variance — all contextual ads have a null ad_LocationID, so the feature carries no information. Replaced with category_match.

---

## 5. Challenges and How I Addressed Them

**Leakage risk in cumulative user features.** user_impression_count and user_click_count must represent what the model would have known *before* the current impression occurred, not including it. A naive cumulative sum would include the current row and leak future click information. I sorted the dataframe by `UserID, SearchDate` and applied `groupby().cumsum()` followed by `shift(1)`, so each row's value reflects only prior observations for that user. This is the same approach as the Gzsiceberg KDD Cup pipeline.

**location_match zero variance.** I expected to build a location match feature (does the ad's location match the user's search location?) analogous to category_match. During engineering I discovered that all contextual ads in this dataset have a null `ad_LocationID`. The feature would be zero for every row, adding no information and potentially confusing tree models. I documented this, discarded it, and substituted category_match, which had meaningful variance.

**Smoothing parameter transfer.** The rate encoding uses α=0.05, β=75 — the same values tuned in KDD Cup NB02. I transferred these without re-tuning on Avito data. The justification is that both datasets have similar data sparsity structures (many entities with few observations), and re-tuning would require a separate validation set that was not available at this stage. The parameters are therefore a principled starting point, not an optimised choice.

**NaN landscape.** After all imputation, only price_log had any missing values: 1,789 rows (0.07%) where the ad had no price despite the listing being contextual. These are handled by the has_price=0 flag and imputed to 0 in NB06 before modelling.

---

## 6. Techniques Used

- Chronological sort + `shift(1)` for leakage-safe cumulative features
- Laplace smoothing: `(clicks + α × β) / (impressions + β)`, α=0.05, β=75
- `numpy.log1p` for price transformation (handles zero safely)
- `str.split().len()` for title word count
- Category hierarchy join on `CategoryID → Level`
- Pearson correlation matrix against IsClick for feature selection
- Parquet output for downstream notebooks

---

## 7. What I Would Do Differently

**Re-tune smoothing parameters on Avito.** The α=0.05, β=75 values were tuned for KDD Cup. Avito's CTR distribution (0.6142% vs KDD's 4.2%) and entity frequency distributions differ meaningfully. A small grid search over β on a held-out fold could improve the rate encoding quality without much effort.

**Add ad-image features.** Avito is a classifieds platform where ad images are likely a major driver of click behaviour. This dataset does not include image features, but in a real production system image quality signals (resolution, whether an image exists, image category classifier output) would probably rank highly.

**Engineer query–ad text similarity.** SearchQuery is available in SearchInfo but I treated it as high-cardinality categorical (too many unique queries to encode directly). A proper approach would use TF-IDF or character n-gram similarity between the search query and the ad title to build a relevance score. Given that KDD Cup's pQId (smoothed query CTR) was the second-most important feature, text relevance is likely underweighted in the current feature set.

**Validate session_size binary constraint.** session_size being binary (only 1 or 2) in this dataset may be an artifact of the 5M-row sample rather than a true dataset property. The full Avito dataset may have longer sessions. I would verify this before drawing any conclusions about session depth effects.
