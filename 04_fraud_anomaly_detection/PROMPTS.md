# Prompts — Fraud & Anomaly Detection

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build an anomaly-detection project on a popular real, imbalanced dataset,
same CRISP-DM tab structure as the rest of the portfolio. Don't let accuracy
be the headline metric on an imbalanced problem — use PR-AUC and explain
why. Compare a genuinely unsupervised anomaly detector against supervised
baselines, and give the Evaluation tab a real cost-sensitive angle, not just
a static confusion matrix.
```

## Key follow-up decisions made along the way

* **Dataset**: the Kaggle/ULB "Credit Card Fraud Detection" dataset (the
  canonical imbalanced-classification benchmark) was pulled via a public
  GitHub mirror of the same file (Kaggle's own API requires interactive
  auth) and stratified down from 284,807 to ~14,500 rows — keeping **every**
  confirmed fraud, sampling only the majority class, so the repo stays
  small without ever losing a positive example.
* **True unsupervised anomaly detection, not a label shortcut**: the
  Isolation Forest is explicitly fit only on `X_train[y_train == 0]`
  (never shown a fraud label), rather than fit on the full training set
  with `contamination` set from the known fraud rate — the latter would be
  a subtle form of label leakage into an "unsupervised" model.
* **Cost-sensitive threshold tuning added over a plain confusion matrix**:
  since the business framing explicitly calls out asymmetric costs (a
  missed fraud vs. a false alarm), the Evaluation tab was built around an
  interactive expected-cost curve with sliders for both dollar costs,
  rather than a single fixed 0.5 threshold — this is the direct
  business-facing payoff of choosing PR-AUC as the modeling metric in the
  first place.
* **Deployment tab realism**: "Sample from test set" mode shows the
  ground-truth label so a reviewer can sanity-check the model against known
  cases (both a true fraud and a true legitimate transaction were manually
  verified to score correctly — 100% and 0.5% fraud probability
  respectively — before considering the tab done).

## Verification

Headless Playwright drove all six tabs and additionally exercised the
Deployment tab twice: once on a known-legitimate test-set row and once on a
known-fraud row (row index looked up in a throwaway script against the same
train/test split the app uses), confirming the model's predicted probability
matched the ground truth in both directions before treating the feature as
verified.
