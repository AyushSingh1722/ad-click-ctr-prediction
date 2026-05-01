# KDD Cup 2012 Track 2 — Feature Engineering and Modelling

## Why this notebook exists

The EDA notebook established what the data contained and which variables showed discriminative signal. What it could not do was connect that signal to a model — because the raw data, as delivered, is not learnable. Every entity in the dataset is an anonymised integer. You cannot pass a user ID or a query ID directly into XGBoost and expect it to do anything useful; the cardinality is in the millions and the IDs carry no ordinal meaning. The bridge between the EDA findings and a working model is feature engineering, and specifically the question of how to convert identity into behaviour. That is what this notebook answers.

The core idea is straightforward in retrospect but requires care in execution: instead of feeding raw IDs, compute each entity's historical click-through rate from the training data and use that as a continuous feature. The challenge is that most entities appear too infrequently for a raw rate to be reliable. An advertiser with two impressions and one click has a raw CTR of 50%, which is not a signal — it is noise. Smoothing those estimates toward the global mean is what makes the features usable, and choosing how aggressively to smooth is itself a modelling decision. Once the rate features were in place, model selection became almost secondary: the feature engineering did most of the work.

---

## Dataset

The source is the same seven-file KDD Cup 2012 Track 2 release described in notebook 01: 5 million training impressions and 1 million test impressions, with lookup tables for queries, keywords, ad titles, descriptions, and user profiles. The difference here is what is done with those files.

In the EDA notebook, the lookup tables were joined to compute descriptive statistics. Here, the training impressions are used to build per-entity click history aggregates, which are then merged back as features for both training and test rows. The solution file — which provides click labels for the test set — is used strictly for evaluation after prediction. It is not seen during training or feature construction.

The 5M/1M split is meaningful because any historical CTR feature computed from training data will have coverage gaps on the test set. Test rows can contain user IDs, query IDs, or advertiser IDs that never appeared in training. Those entities get no historical estimate and produce NaN feature values, which need to be handled explicitly before the model can run.

---

## Approach

The feature engineering strategy centres on a family of twelve smoothed rate features, each named with a `p*` prefix to distinguish them from the raw ID columns. For each entity type — user (pUId), query (pQId), ad title (pTitleId), ad description (pDescId), keyword (pKeyId), advertiser (pAdvCTR), ad creative (pAdCTR), display URL (pAdURL), position (pPosCTR), relative position (pRPosition), depth (pDepthCTR), and gender (pGender), and age (pAge) — I aggregated total clicks and total impressions from the training data and computed a smoothed CTR estimate.

The smoothing formula is additive: the smoothed rate for an entity is (clicks + α) / (impressions + β), with α=0.05 and β=75. The β=75 prior means that an entity needs roughly 75 impressions before its observed click rate is trusted at face value. Below that threshold, the estimate shrinks toward α/β — a near-zero prior that reflects the global rarity of clicks. This is conservative by design. The alternative — trusting small-sample estimates — would let a single high-CTR observation from a brand-new advertiser pollute the feature space. I started with β=75 based on the advertiser frequency analysis in notebook 01, where the divergence between median and mean CTR for infrequent advertisers made the noise problem concrete.

After computing the p* features from training data, I merged them onto both the training and test rows by joining on the entity IDs. For test rows where an entity ID had never appeared in training, the join produced NaN values. I imputed these with the global smoothed CTR from the training set — the same prior the smoothing formula uses — rather than with zero or the column mean. Using zero would tell the model that an unseen entity has no historical clicks, which conflates absence of data with evidence of poor performance. Using the global prior is the honest answer: we have no data on this entity, so we fall back to the base rate.

I dropped pDepthCTR from the final feature set after initial experiments showed it added no lift over the other position-related features already in the model. The final model trains on twelve p* features.

Grid search covered combinations of depth (2 and 4) and n_estimators (200, 400, 800) at a fixed learning rate of 0.01. I held learning rate fixed at a low value deliberately — with 800 estimators and depth 2, slow learning with many rounds consistently outperforms fast learning with fewer rounds on problems where the individual features are already smooth aggregates rather than raw sparse inputs. The winning configuration was depth=2, n_estimators=800, learning_rate=0.01.

Evaluation uses a custom AUC implementation rather than sklearn's `roc_auc_score`. Sklearn's implementation expects binary labels (0 or 1), but the KDD Cup training data provides raw click counts and impression counts per row — click counts can be greater than 1 if the same impression was recorded multiple times. Passing multi-valued targets directly into sklearn's AUC produces silently wrong results. The custom implementation handles the click/impression count format correctly by deriving the binary outcome from whether any clicks occurred, without collapsing through sklearn's assumption about the label space.

---

## Key Findings

1. **Replacing raw entity IDs with smoothed historical CTR rates transformed an unlearnable problem into one that achieves AUC=0.6803.** Raw integer IDs have no numerical meaning to a gradient-boosted tree — splitting on user ID 4,823,001 versus 4,823,002 is arbitrary. Smoothed historical CTR converts those IDs into a continuous signal that directly encodes past click behaviour, giving the model something meaningful to split on. The gap between a model with rate features and a model with raw IDs is not incremental; it is the difference between a working model and a null result.

2. **Shallow trees (depth=2) outperformed deeper ones (depth=4), reaching AUC=0.6803 versus 0.6728.** This is the expected outcome given the nature of the p* feature set. The features are already smoothed aggregate statistics — each one is a single continuous value encoding an entity's click history. There is limited combinatorial interaction structure for a deep tree to exploit; a depth-2 tree can already capture the first-order interactions (e.g., high-pUId and high-pPosCTR together). Deeper trees found spurious splits in the training data that did not generalise, and the 0.0075 AUC penalty for depth=4 reflects that directly.

3. **pUId (weight=856) and pQId (weight=655) are the two dominant features, with pTitleId (weight=436) a clear third — but weight and gain tell different stories.** Feature importance by weight counts split frequency, which favours features the model returns to repeatedly in shallow trees. Gain importance, which measures average loss reduction per split, shows pTitleId gaining more ground per use than its split count alone suggests. The practical implication is that the ad creative — specifically its title — carries more predictive content than split count implies. The user and query together define the context of the impression; the title is where ad relevance enters.

4. **pAdCTR, pGender, and pAge all recorded zero feature importance in the final model, confirming what the EDA signalled.** pAdCTR — the historical CTR of the specific ad creative — turns out to be redundant once pTitleId and pDescId are in the model; those features already capture the creative-level signal through the title and description. pGender and pAge being zero is the model's confirmation of the flat demographic splits from the EDA: Male=4.41%, Female=4.59%, Unknown=4.53% in click rates, and an age pattern that was non-monotonic. The model found nothing to split on. Including these features in the grid search added no cost, but their zero importance is a clear result, not just a soft finding.

5. **Without smoothing, low-impression entities would have unstable rate estimates that corrupt the model.** The β=75 prior is the mechanism that makes the features usable. An entity with five impressions and three clicks has a raw CTR of 60% — a figure that would make it look like the most clickable entity in the dataset, when in reality the sample is too small to mean anything. With β=75, the same entity's smoothed estimate is 3.05/80 ≈ 3.8%, just above the global mean, which is the honest answer given the evidence. Every p* feature depends on this smoothing to be well-behaved, and the α and β values are the single most consequential hyperparameter decision in the entire pipeline.

---

## Challenges and How I Addressed Them

**Seven-file merge fragility on both training and test.** The feature construction pipeline joins the training impression table against six lookup tables, aggregates by entity ID, and then joins back to both training and test rows. Any mismatch in join keys — wrong column names, mismatched dtypes between the impression table and lookup table IDs — produces silent NaN inflation or, worse, a row count change that only becomes visible downstream. I validated the row count and NaN rate at each join step rather than checking only at the end, which caught two dtype mismatches early.

**NaN imputation for test-set unseen entities.** Roughly 5–10% of test rows contained entity IDs not seen in training, producing NaN values in one or more p* features. The choice of imputation value matters: zero implies zero historical click rate, which is not the same as unknown. I imputed with the global smoothed prior computed from training data, which reflects genuine uncertainty rather than poor performance. Features with higher NaN rates on the test set (pUId is the most volatile, since individual users appear infrequently) have softer predictive power on unseen users, and the model's relatively low weight on pAdvCTR (1) versus pUId (856) reflects this — advertisers recur across many impressions, while individual users do not.

**Sklearn's AUC silently producing wrong results.** The click column in the training data is a count, not a binary label. Passing it directly to `roc_auc_score` without converting to binary first produced plausible-looking AUC values that were technically incorrect. This is the kind of bug that is hard to catch because the output is not obviously wrong — the AUC was in a reasonable range, just not the right one. I wrote a custom evaluation function that explicitly derives a binary click indicator from the click count and impression count before computing AUC, which made the evaluation logic transparent and verifiable.

**XGBoost silently ignoring the `silent` parameter.** In the version of XGBoost used here, the `silent=True` parameter was deprecated but accepted without error — the model simply printed verbose output regardless. This was a minor irritant rather than a correctness issue, but it illustrates a real problem with XGBoost's API surface: deprecated parameters fail quietly rather than with a clear warning. The fix is `verbosity=0`, not `silent=True`, but you only discover this by reading the changelog.

---

## Techniques Used

A note on naming: every engineered feature in this notebook carries a p* prefix — pUId, pQId, pTitleId, and so on. The p stands for pseudo-CTR. The raw columns are anonymised integer IDs (UId, QId, TitleId) that carry no numerical meaning on their own. The p* features are what those IDs become after the transformation pipeline: each entity's smoothed historical click-through rate, derived from aggregating its clicks and impressions across the training data. The prefix is a deliberate marker that distinguishes "this column is a rate estimate derived from an ID" from "this column is a raw ID." It also makes the feature importance output readable at a glance — seeing pUId=856 in the importance plot immediately tells you this is the user's historical CTR, not a user identifier.

**Smoothed rate features (pseudo-CTR encoding):** Raw categorical IDs have no numerical meaning to a tree model; converting them to historical CTR estimates gives the model a continuous signal that directly encodes past behaviour. Smoothing with α=0.05 and β=75 is necessary because low-impression entities would otherwise produce extreme and unreliable rate estimates.

**Additive smoothing with a fixed prior:** Bayesian smoothing toward a global mean prevents small-sample noise from dominating the feature space. The β=75 prior was chosen based on the EDA finding that advertiser click rates become stable only after a substantial number of impressions — it is the mechanism that makes the p* features well-behaved across the full entity frequency distribution.

**XGBoost regressor rather than classifier:** The target is a probability of click, not a hard binary label. Using a regressor with the training click counts directly (after appropriate normalisation) allows the model to produce calibrated probability outputs, which is what the AUC and MAPE evaluation metrics require. A binary classifier would have discarded information about impression weight.

**Custom AUC evaluation:** Necessary because sklearn's `roc_auc_score` assumes binary integer labels, while the KDD Cup data provides raw click counts and impression counts. Writing the evaluation explicitly made the metric computation auditable and prevented the silent misalignment that occurs when you pass count data to a function expecting binary flags.

**Grid search over depth and n_estimators at fixed low learning rate:** With a small learning rate (0.01), many estimators are required to converge — but this combination consistently generalises better than fewer estimators at a higher rate. I fixed learning rate and searched over the structural parameters (depth, n_estimators) because the interaction between depth and overfitting is the key uncertainty on a dataset with smooth aggregate features.

---

## What I Would Do Differently

**Use a time-based train/test split instead of random.** Random splitting means that when I compute historical CTR features for a training row, I may be including impressions that occurred after the test set's impressions in time. This is a mild form of data leakage — the feature for a test-period impression incorporates information from the future. A proper evaluation splits by timestamp, training on all impressions before a cutoff date and testing on those after. The KDD Cup data includes timestamp information; I did not use it for splitting. The AUC of 0.6803 is likely optimistic by a small amount because of this.

**Search over the smoothing parameters α and β.** I chose β=75 based on the advertiser frequency analysis in notebook 01, which gave a principled starting point but was not a systematic search. Different entity types probably benefit from different β values — a user ID that appears twice and an advertiser that appears ten thousand times have very different sample sizes, and a single global β treats them identically. Separate smoothing parameters per entity type, or even a cross-validated search over β, would likely improve the features for low-frequency entities.

**Evaluate feature importance by gain, not just weight.** I reported importance by weight (split count) because it is the default and most legible output, but gain importance is usually more informative about which features actually reduce loss. The finding that pTitleId carries more per-split gain than its weight ranking suggests — despite being third by weight — is a result I only noticed after the modelling was done. Starting with gain importance would have shaped the feature discussion differently and might have prompted earlier investigation into the title features.
