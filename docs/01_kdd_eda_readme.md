# KDD Cup 2012 Track 2 — Exploratory Data Analysis

## Why this notebook exists

Click-through rate prediction at industrial scale is a different problem from most supervised learning tasks you encounter in coursework or small portfolio projects. The signal is sparse, the data is anonymised, and the baseline — just predict the global average — is surprisingly hard to beat cleanly. I chose the KDD Cup 2012 Track 2 dataset specifically because it captures this messiness honestly: no readable text, no interpretable user profiles, just hashed IDs and historical interaction counts. Before building any model, I needed to understand what the data actually contained and where the predictive signal was likely to live.

This EDA notebook is the foundation for the modelling work in notebook 02. The decisions made here — which variables showed discriminative power, which were nearly flat, how skewed the target was, and how ad position related to click rate — shaped every feature engineering choice that followed. An EDA that stops at shape and missingness is not much use; this one was specifically trying to answer which entity-level signals (user, query, advertiser, position) were worth the effort of engineering smoothed historical CTR features for.

---

## Dataset

The KDD Cup 2012 Track 2 release consists of seven files that must be joined before any useful analysis can begin:

- **training.txt** — the main impression log, 5 million rows. Each row is one ad impression with columns for click count, impression count, display URL, ad ID, advertiser ID, depth (total ads shown in the session), position (rank of this ad), query ID, keyword ID, title ID, description ID, and user ID.
- **test.txt** — 1 million held-out impressions in the same format, without click labels.
- **queryid_tokensid.txt** — maps each query ID to a sequence of token IDs representing the search query terms. The tokens themselves are also integers; the raw query text is not provided.
- **purchasedkeywordid_tokensid.txt** — maps keyword IDs to token sequences. These are the advertiser-purchased keywords that triggered the ad.
- **titleid_tokensid.txt** — maps ad title IDs to token sequences.
- **descriptionid_tokensid.txt** — maps ad description IDs to token sequences.
- **userid_profile.txt** — maps user IDs to gender (1=Male, 2=Female, 0=Unknown) and age group (1–6, where 1 is under 18 and 6 is the oldest bracket).

After merging all seven files on their respective ID keys, the working dataset contains 4.95M rows. The target variable is binary click/no-click, derived from the click count and impression count columns in the training file.

The most important quirk of this dataset is that almost nothing in it is human-readable. All entity IDs are anonymised integers. The token sequences in the lookup tables let you compute token-level features (query length, title length, description length) but not semantic ones. You cannot read what a query says or what an ad title contains — only how many words each has and which token IDs appear. This is a meaningful constraint on what EDA can reveal.

---

## Approach

I worked through the data in four passes, each answering a progressively more specific question.

The first pass was structural. I loaded the training file, checked dtypes, confirmed there were no missing values in the core impression columns, and computed the global CTR from the click and impression counts. This gave me the 4.2% baseline and confirmed the target distribution was heavily right-skewed — the vast majority of impressions record zero clicks, and a small number of high-performing ads pull the mean up from zero.

The second pass focused on the position and depth variables, which are the two structural controls the ad platform has over any given impression. Position (1, 2, or 3) and depth (how many ads appear in total) are observable and categorical, which made them natural starting points before tackling the noisier entity-level signals.

The third pass joined the user profile table and looked at demographic signals: gender and age group. I computed CTR broken out by each demographic segment and compared means. Given how flat the demographic splits turned out to be, I spent additional time checking whether this was a data quality issue (many users in the Unknown gender category) or a genuine signal absence. The conclusion was signal absence — the variation within demographic groups was larger than the variation between them.

The fourth pass brought in the query, title, and description lookup tables to compute text length features. Rather than working with token IDs directly, I counted the number of tokens per entity, which produced interpretable numeric features. I then looked at how CTR varied across query length, title length, and description length buckets. Finally, I aggregated by advertiser ID to examine how CTR distributions differed between frequent and infrequent advertisers, which directly motivated the smoothing approach used in notebook 02.

---

## Key Findings

1. **The net CTR of 4.2% is the only honest baseline, and the target distribution is almost entirely zeros.** Most impressions record no clicks at all; the distribution is so right-skewed that mean CTR and median CTR diverge sharply. Any model that learns to predict near zero for every impression will look reasonable by MSE but will be completely useless in practice. This pushed me toward AUC as the primary evaluation metric in notebook 02.

2. **Ad position creates a near-3x gradient in click rate: Position 1 at 5.52%, Position 2 at 3.08%, Position 3 at 1.89%.** This is not subtle. Position is load-bearing — it almost certainly carries more predictive weight than any demographic feature. The implication is that a model trained without position information is missing a dominant signal, and that position's CTR effect needs to be encoded as a feature rather than treated as a confound to control away.

3. **CTR peaks for 3-word queries at 4.90% and falls off at both shorter and longer queries.** Very short queries are probably navigational — the user knows where they are going and is unlikely to click an ad. Very long queries are probably highly specific research queries where the ads shown are less relevant by the time the user has narrowed their intent that precisely. Three words appears to be the sweet spot where ad relevance and user purchase intent align. The 75th percentile of query length is 4 words, so the peak sits within the bulk of the distribution, not at an edge.

4. **The age pattern reverses where intuition would predict otherwise: the 18–30 cohort (groups 3 and 4) clicks less than the under-18 cohort, and the 30+ groups have the highest CTRs at 4.75% (group 5) and 5.26% (group 6).** One explanation is that younger users are more ad-literate and have learned to ignore search ads. An alternative is that the 18–30 group skews toward information queries while the 30+ group skews toward transactional ones. I cannot distinguish between these explanations from the data, but the pattern mattered — it warned me not to assume age was monotonically predictive in either direction, and it aligned with the model's eventual finding that age contributed zero feature importance.

5. **Frequent advertisers have a median CTR of 3.51% versus 1.58% for infrequent ones, but the mean gap is much smaller (4.14% vs 3.89%).** Frequent advertisers are more consistent — they show up repeatedly because they have learned which keywords and placements work, and their click rates cluster around a reliable middle. Infrequent advertisers are more bimodal: a few perform very well, most perform poorly, and the high variance means a simple mean hides what is actually happening. This directly motivated using the 75th percentile (5.13%) rather than the mean as the threshold for classifying high-CTR advertisers, and it shaped the β=75 smoothing prior used in notebook 02.

---

## Challenges and How I Addressed Them

**Seven-file merge complexity.** The dataset is intentionally split across multiple lookup tables to minimise file size, but reassembling it correctly requires joining on multiple keys without introducing duplicates or nulls. I worked through the joins sequentially — training to user profile first, then layering in query, keyword, title, and description token counts — and validated row counts at each step to confirm the merge was clean before proceeding to analysis.

**Memory pressure on 4.95M rows.** A full join of all seven tables produces a wide dataframe that pushes memory limits on a standard laptop environment. I used dtype optimisation (downcasting integers where possible) and avoided materialising intermediate wide tables that were not needed downstream. Visualisations were computed on aggregated summaries rather than the raw row-level data.

**Visualising extreme CTR skew.** The target distribution is so right-tailed that a standard histogram shows a spike at zero and almost nothing else. I applied log-scaling on the y-axis and used separate plots for the full distribution versus the non-zero click subset to make the shape legible. For entity-level CTR (e.g., advertiser CTR distribution), I used box plots rather than histograms because the variance within groups was as informative as the central tendency.

---

## Techniques Used

**Grouped aggregation by categorical variables (position, depth, gender, age):** The fastest way to assess whether a categorical feature has discriminative power is to group by it and compare mean CTR across groups. I used this before any modelling to build intuition about which variables were worth engineering features from.

**75th-percentile thresholding for advertiser classification:** Mean-based splits are distorted by high-variance infrequent advertisers; the 75th percentile of advertiser CTR (5.13%) gives a more stable threshold that separates genuinely high-performing advertisers from noise. I used this to label advertisers rather than treating advertiser CTR as a raw continuous feature.

**Token count as a proxy for text length:** Since the raw text is not available, counting the number of token IDs per query, title, or description gives a noisy but usable signal for text length. This is a coarse proxy — two queries with the same word count can be semantically very different — but it is the only text-level feature derivable from anonymised token sequences.

**Log-scale visualisation for skewed distributions:** Applied to the CTR histogram and to impression count distributions, where a linear y-axis collapses all meaningful variation into a spike at the low end. Log scaling makes the tail structure visible without distorting the interpretation.

---

## What I Would Do Differently

**Investigate the Unknown gender category more carefully.** A large share of users fall into the Unknown (0) gender category, and I largely treated it as a third level of the gender variable rather than asking why it exists. Users may be Unknown because they did not register with the platform, because their profiles lacked that field, or because of data collection differences across time periods. These are meaningfully different populations that might have different click behaviour for structural rather than demographic reasons.

**Compute CTR at the query-advertiser intersection, not just separately.** I looked at query CTR and advertiser CTR as independent dimensions, but the interesting signal is probably in their interaction — certain advertisers may perform well on specific query types and poorly on others. Doing this properly requires enough data per cell to get stable estimates, which is exactly the smoothing problem addressed in notebook 02, but even a rough version of this cross-analysis in EDA would have given earlier intuition about the feature space.

**Profile the non-clicked impressions separately.** Most of the EDA treats impressions as the unit of analysis and click rate as the outcome. It would have been more informative to split the dataset into clicked and non-clicked subsets and profile each independently — looking at which positions, query lengths, and advertiser types were over-represented in the clicked set versus the overall impression pool. That framing would have made the discriminative structure more visible earlier in the process.
