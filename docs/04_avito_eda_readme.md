# Notebook 04 — Avito CTR Prediction: Exploratory Data Analysis

## 1. Why This Notebook Exists

Phase 1 (NB01–03) worked with the KDD Cup 2012 dataset. Before writing a single feature for the Avito dataset, I needed to answer three questions: Is this dataset structured the same way? Do the same behavioural patterns hold? Is there anything that would silently corrupt a model if left unaddressed?

NB03 showed that model choice is data-dependent — logistic regression tied LightGBM on the small advertising dataset. That result pushed me to explore Avito carefully rather than assuming the KDD Cup playbook would transfer. The IsContext artifact (described below) was the first thing that would have ruined everything if I had skipped this step.

---

## 2. Dataset

**Avito Context Ad Clicks 2015** (Kaggle competition dataset). Six files merged across common keys:

| File | Rows | Description |
|------|------|-------------|
| trainSearchStream | 5,000,000 | Core event log: SearchID, AdID, Position, HistCTR, IsClick |
| SearchInfo | 1,370,000 | Search context: date, user, location, category, query |
| AdsInfo | 1,960,000 | Ad attributes: category, location, price, title, IsContext |
| UserInfo | 4,280,000 | Device and user-agent metadata |
| Category | 68 | Category hierarchy with levels |
| Location | 4,080 | Location hierarchy with region/city |

**Working set after filtering:** 2,422,983 rows where `IsContext == 1`.

---

## 3. Approach

I merged all six files on their natural keys (SearchID → SearchInfo, AdID → AdsInfo, UserID → UserInfo, CategoryID → Category, LocationID → Location), then performed univariate and bivariate analysis on every available signal before touching any feature engineering.

The sequence was:
1. Identify and document the IsContext artifact before any analysis
2. Compute click rates by position, category match, login status, and price quantile — all on the contextual-only subset
3. Score HistCTR as a single-feature baseline using AUC
4. Compare every finding against the KDD Cup equivalents to identify what transfers and what differs

All analysis from this point forward uses only the 2,422,983 contextual rows.

---

## 4. Key Findings

**Click rates:**
- Contextual click rate (IsContext=1): **0.6142%** — the true click rate for ads we care about
- Raw click rate across all 5M rows: **0.2977%** — misleading; includes 2,577,017 IsContext=0 rows with structural zero clicks
- IsContext=0 rows: **2,577,017** — these all have IsClick=0 by dataset construction, not user behaviour

**HistCTR baseline:**
- AUC with HistCTR as a single feature: **0.6640**
- This is competitive: KDD Cup required 12 engineered features to reach AUC=0.6803
- Avito provides a pre-computed rolling CTR, which KDD Cup did not — this shifts the problem from feature engineering to calibration

**Position gradient:**
- Pos1=**0.7309%**, Pos7=**0.4614%**, ratio≈**1.6×**
- KDD Cup equivalent: Pos1=5.52%, Pos3=1.89%, ratio≈2.9×
- Avito's position effect is substantially weaker — ads here are more homogeneous in quality than search ads

**Category and login signals:**
- Category mismatch CTR: **0.3167%** vs match CTR: **0.2921%** — mismatched ads are clicked more
- Logged-out CTR: **0.3180%** vs logged-in CTR: **0.2553%** — logged-out users click more
- Both findings run counter to what I expected, suggesting confounds worth investigating in feature engineering

**Price:**
- CTR broadly increases with price — higher-priced ads earn slightly more clicks
- No step-function breaks; the relationship is gradual

---

## 5. Challenges and How I Addressed Them

**The IsContext artifact.** The raw dataset has 5M rows but 2.577M of them have IsContext=0 and zero clicks structurally — not because users didn't click, but because these rows were never eligible for clicking. A model trained on raw data would learn that most impressions have zero clicks and bias all predictions toward zero. I discovered this by comparing the raw click rate (0.2977%) against the contextual click rate (0.6142%) and tracing the gap to the IsContext column. The fix was to filter to IsContext=1 before all analysis and modelling.

**HistCTR as a feature vs a baseline.** Avito provides a pre-computed rolling historical CTR for each ad. This did not exist in the KDD Cup dataset — there we built it from scratch. I had to decide whether to treat it as a raw feature (which respects whatever smoothing Avito applied) or rebuild it with our own smoothing. I chose to treat it as a feature and separately compute our own smoothed CTRs in NB05, giving the model both signals and letting it weight them.

**Counterintuitive directional findings.** Category mismatch and logged-out users both showing higher CTR than their respective baselines required careful verification. I checked both against multiple subsets to confirm they were not sampling artifacts. Both held. I documented them as potential confounds to revisit in feature importance analysis rather than engineering them out.

---

## 6. Techniques Used

- Pandas multi-file merge on foreign keys
- Stratified click rate calculation by category, position, login status, and price quartile
- Single-feature AUC evaluation (`sklearn.metrics.roc_auc_score`) for the HistCTR baseline
- Bar charts and line plots for CTR-by-segment comparisons
- Cross-reference against KDD Cup findings to measure signal transfer

---

## 7. What I Would Do Differently

**Build a proper data quality report first.** I discovered the IsContext artifact during EDA, which was lucky. A production workflow would run a data contract check — expected row counts, expected null rates, expected click rate range — before any analysis begins.

**Dig further into the counterintuitive findings.** The logged-out > logged-in CTR and the category mismatch > match CTR findings are documented but not explained. I would add user-level cohort analysis to test whether logged-out users in this dataset are a specific subpopulation (e.g., new users visiting from search engines) that explains the directional reversal.

**Profile HistCTR quality.** HistCTR is a black-box pre-computed feature from Avito. I used it as-is without auditing how it was computed — how many impressions it is based on, whether it includes non-contextual rows, or how it handles new ads. A production system would need to understand and replicate that computation exactly.
