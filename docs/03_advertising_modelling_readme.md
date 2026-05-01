# Advertising Dataset — EDA and Model Comparison

## Why this notebook exists

The two KDD Cup notebooks deal with a specific kind of complexity: five million rows of anonymised data where the features carry no semantic meaning and the entire challenge is engineering a learnable representation from raw IDs. The results there — gradient boosting on smoothed rate features, AUC 0.6803 — say something about what works when data is noisy, sparse, and requires heavy preprocessing. They do not say anything about whether gradient boosting is the right tool in general. That question needs a different dataset to answer.

This notebook takes a small, clean advertising dataset — 1,000 rows, 9 features, no missing values, all human-readable — and runs four models against it side by side. The purpose is not to squeeze out a higher accuracy; on a dataset this clean, several models will plateau near the same ceiling. The purpose is to see whether the model hierarchy from the KDD Cup work holds when the data no longer demands heavy feature engineering. The short answer is that it does not — and that reversal is the finding worth documenting.

---

## Dataset

The dataset is `data/sample/ad_click_data.csv`: 1,000 rows and 9 columns with no missing values. Every feature is interpretable without a lookup table. The columns cover:

- **Daily Time Spent on Site** — minutes per session, continuous
- **Age** — user age in years, continuous
- **Area Income** — estimated local income, continuous
- **Daily Internet Usage** — minutes of total daily internet use, continuous
- **Ad Topic Line** — the ad headline, text (not used as a model feature, kept for reference)
- **City** — user's city, categorical (high cardinality, not used as a model feature)
- **Male** — binary gender indicator
- **Country** — user's country, categorical (not used as a model feature)
- **Clicked on Ad** — binary target: 1 if the user clicked, 0 if not

The train/test split is 75/25: 750 rows for training and 250 rows for testing. The class balance is close enough to even that no resampling was needed — the 250-row test set contains 137 non-clicks and 113 clicks, so both classes are well-represented in evaluation.

The most important characteristic of this dataset, and the one that shapes every modelling decision, is that the features have direct semantic meaning. Time spent on site, age, and area income are variables a human can reason about without any encoding or aggregation. This is the opposite of the KDD Cup setting, where the features had to be constructed from scratch before the model could do anything with them.

---

## Approach

I started with a brief EDA before touching any model. With only nine features and no missing values, the EDA was primarily about understanding feature distributions and their relationship to the target — which features separated clickers from non-clickers most cleanly, and whether the separation looked linear or required more complex boundaries.

The continuous features showed clear bivariate separation. Daily internet usage and daily time on site had opposite patterns for clickers and non-clickers: users who clicked tended to spend less time on site and use the internet less overall, while non-clickers were heavier internet users. This is a counterintuitive pattern — heavy internet users apparently have higher ad fatigue or are more skilled at ignoring ads. Area income showed a similar clean split, with lower-income users clicking more. Age skewed younger for clickers. These patterns suggested a dataset where a well-fitted linear boundary could do serious work, which set the expectation before any model was run.

For modelling, I ran four classifiers in sequence: Logistic Regression, LightGBM, XGBoost, and a Decision Tree. All four used default or lightly tuned hyperparameters — the point was comparison, not squeezing the last fraction of a percent from any individual model. I evaluated each on the held-out 250-row test set, reporting accuracy and the full confusion matrix so the type-I and type-II error rates were visible, not just the headline number.

The categorical columns (City, Country, Ad Topic Line) were dropped rather than encoded. City has high cardinality and 1,000 rows; one-hot encoding it would have produced a sparse matrix with more columns than rows, and the signal from city would almost certainly be noise at this sample size. Country has similar issues. Dropping them was the correct call given the dataset size.

---

## Key Findings

1. **Logistic Regression tied LightGBM at 97.2% accuracy (243 of 250 correct) — the exact opposite of the KDD Cup result.** On the KDD Cup data, LR would have had almost nothing to train on without the p* features, and even with them, gradient boosting was the right tool for capturing non-linear interactions in a sparse, high-dimensional feature space. On this dataset, the features are continuous, low-dimensional, and the decision boundary is approximately linear. LR found it cleanly. The reversal is not a surprise once you look at the data; it is exactly what you would expect when the distributional assumptions behind LR are roughly met.

2. **XGBoost came in at 94.8% (237/250), 2.4 points below LightGBM despite being the stronger model on the KDD Cup task.** With 750 training rows and four continuous features, XGBoost has very little room to do what it does well — find complex, high-order interactions across many features. Its default configuration likely overfit slightly, and the tree structure it built was more complex than the data warranted. LightGBM's different splitting strategy and regularisation defaults happened to generalise better here, though the gap between them on a different random seed might narrow or flip.

3. **The confusion matrix for LR (TN=132, TP=111, FP=5, FN=2) shows the errors are not symmetric.** LR produced only 2 false negatives — cases where a user would have clicked but the model predicted no click. False positives (5 cases) were slightly more common. In an advertising context, these error types have different costs: a false negative means a missed impression opportunity; a false positive means a wasted placement. Neither is severe at this accuracy level, but the asymmetry is worth noting for any downstream decision about whether to optimise the classification threshold.

4. **The Decision Tree was the weakest model at 94.4% (236/250), which confirms the dataset is not tree-friendly at this depth.** A single decision tree overfits easily on small datasets and cannot compensate with ensemble methods the way gradient boosting does. At 750 training rows, a moderately deep tree will find splits that are specific to training noise rather than genuine patterns. The 2.8-point gap between the Decision Tree and LR is the clearest signal that model complexity without regularisation is the wrong direction on data this size.

5. **All four models clustered between 94.4% and 97.2% — the ceiling is the dataset, not the model.** When the lowest-performing model (Decision Tree) is within 3 points of the best, you are not in a regime where model selection drives outcomes. The features have strong signal, the classes are well-separated, and any reasonable classifier will find the boundary. The lesson from this clustering is that investing more effort in feature quality and data collection would have a much higher return than switching between gradient boosted implementations.

---

## Challenges and How I Addressed Them

**Variance in accuracy metrics at 1,000 rows.** A single 75/25 split means the 250-row test set is the entire basis for comparison. At this scale, a difference of 6 correct predictions is the gap between 94.8% and 97.2%. That gap could plausibly reverse on a different random seed. I reported it honestly rather than treating it as a definitive ranking, and the confusion matrices gave enough structure to make the comparison meaningful beyond a single number.

**High-cardinality categorical features with too few rows to encode usefully.** City (potentially hundreds of levels) and Country both appeared in the dataset. One-hot encoding either would produce a matrix wider than it is tall on the training set, guaranteeing overfit for any tree model and multicollinearity problems for LR. I dropped them rather than engineering coarser groupings, which would have introduced its own arbitrary decisions. With a larger dataset, country-level groupings by click rate would be worth trying.

**Choosing what to compare when all models perform well.** When accuracy is compressed between 94% and 97%, the confusion matrix becomes more informative than the headline number. I shifted the focus of comparison from accuracy to error type — which models produced false negatives versus false positives, and what those errors mean in an advertising context — rather than treating the 2.4-point gap between XGBoost and LR as a clean verdict.

---

## Techniques Used

**Logistic Regression as the primary baseline:** LR is the right first model on low-dimensional continuous data where the decision boundary might be linear — not because it is always competitive, but because it is the clearest signal of whether the data has linear structure. If LR performs well, the task is solved and gradient boosting adds complexity without benefit. Here, LR matched the best gradient boosted model exactly.

**LightGBM and XGBoost for gradient boosting comparison:** Running both rather than just one was a deliberate choice. They share the same theoretical foundation but differ in splitting strategy, regularisation defaults, and how they handle small datasets. On a 750-row training set, those implementation differences matter more than on large data, and the 2.4-point gap between them (97.2% vs 94.8%) is a concrete illustration of that.

**Decision Tree as an interpretability anchor:** A single decision tree is rarely the best model, but it produces a structure you can read. Including it in the comparison gives a baseline for what happens when you have no regularisation and no ensembling — the 94.4% result is the cost of that simplicity on data where the boundary is not axis-aligned.

**Confusion matrix analysis over accuracy alone:** Accuracy is a single number that treats all errors as equivalent. The confusion matrix separates false positives from false negatives, which have different consequences in any applied setting. On a balanced binary target, accuracy and the confusion matrix tell similar stories — but the matrix makes the error asymmetry visible in a way the headline number cannot.

**Dropping high-cardinality categoricals rather than encoding:** With 1,000 rows, encoding City or Country produces more columns than the training set can support without severe overfit. Dropping them is a principled decision given the sample size, not just laziness. The four continuous features carry enough signal that the dropped columns added noise rather than information.

---

## What I Would Do Differently

**Use cross-validation instead of a single held-out split.** A single 75/25 split on 1,000 rows gives one draw from a noisy sampling distribution. The 2.4-point gap between LR and XGBoost — six correct predictions — could plausibly shrink to zero on a different random seed. Five-fold cross-validation would have given confidence intervals around each model's accuracy, making the comparison honest about uncertainty rather than presenting single-seed results as a ranking.

**Tune XGBoost's regularisation parameters before comparing.** XGBoost's default configuration is calibrated for larger datasets. At 750 training rows, the default tree complexity likely overfit, and reducing max depth and increasing the minimum child weight would have been worth trying before concluding that XGBoost is weaker than LightGBM here. As presented, the comparison mixes model quality with default hyperparameter choices, and those are not the same thing.

**Compute feature importance for all four models and compare them directly.** The EDA suggested that daily internet usage, time on site, area income, and age were the cleanest separators — but I did not verify whether all four models agreed on that ranking. LR coefficients, LightGBM gain importance, and XGBoost gain importance would have made a more complete picture of which features each model relied on, and any disagreement between them would have been worth investigating. That analysis would have strengthened the narrative about why LR succeeds here rather than just reporting that it does.
