# ML---Business-Classification-Problem
Supervised &amp; unsupervised ML on insurance data. Classifying applicants as high or low claim with Decision Tree & Random Forest, plus K-Means policyholder segmentation.

# HealthShield Insurance — High Claim Prediction & Policyholder Segmentation

Machine learning project on 8,000 health insurance policyholder
records, identifying high-value claimants that a rules-based underwriting
system currently misses.

## Overview
- **Problem:** Supplement rules-based underwriting by predicting whether an
  applicant is likely to be a High Claim case, enabling application triage
  and premium tiering.
- **Supervised:** Binary classification of `High Claim` (3:1 class imbalance
  handled via class weighting). Compared Decision Tree (base /
  pre-pruned / post-pruned) vs Random Forest. **Final model: Random Forest (F1 = 0.52)**.
- **Unsupervised:** K-Means segmentation (excluding target to avoid leakage)
  to surface actionable policyholder groups (e.g. high-risk diabetic segment).

## Workflow
Data cleansing (missing values handled by distribution: skewed numerical
features imputed with median, categorical features with mode; columns with
>50% missing dropped to avoid introducing noise from over-imputation)
→ EDA (univariate / bivariate / multivariate; correlation analysis for
numerical features, Chi-Square tests for categorical features)
→ encoding & transformation (ordinal / dummy coding) → imbalance handling
→ model training & tuning (GridSearchCV, cost-complexity pruning)
→ 5-fold CV → evaluation for two models → K-Means clustering.


## Tech
Python · pandas · scikit-learn · imbalanced-learn · seaborn · matplotlib

## Key Findings
- Hospital Cover Tier is the dominant predictor of high claims.
- Post-pruned DT (5 nodes, depth 2) offers strong interpretability for
  underwriter use; Random Forest chosen for slightly better F1.
- K-Means reveals distinct risk segments for targeted preventive programs.
