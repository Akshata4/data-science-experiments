# Prompts — AutoML Model Tournament

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build an AutoML-style project illustrating automated model selection across
multiple data science tasks, same CRISP-DM tab structure as the rest of the
portfolio. The source project used AutoGluon — if that's too heavy a
dependency for this portfolio, make an honest, transparent scope call and
say so directly in the README rather than silently doing something
different. Cover both classification and regression, tune hyperparameters
automatically, and produce a real leaderboard.
```

## Key follow-up decisions made along the way

* **AutoGluon → scikit-learn scope decision, stated explicitly**: rather
  than install AutoGluon (a large, slow-to-install dependency pulling in
  its own PyTorch/LightGBM/CatBoost stack) or quietly pretend to use it,
  the app was built as a genuine from-scratch AutoML loop
  (`Pipeline` + `RandomizedSearchCV` across 5-6 algorithm families) and the
  README says so up front, in its own labeled section, rather than burying
  the substitution.
* **Real, zero-download datasets**: rather than source a fourth external
  dataset, the three tasks use scikit-learn's own bundled real-world data
  (Breast Cancer, Wine, Diabetes) — genuinely real, well-documented, and
  guaranteed to load with no network dependency, which also makes the CI/
  local-run story simpler for anyone cloning the repo.
* **Search-cost transparency**: real AutoML tools always report a
  time/quality trade-off, so a "search cost per model" chart was added to
  the Modeling tab specifically to make that trade-off visible rather than
  showing only the final ranked score.

## Verification

Every tab was driven with headless Playwright across **all three tasks**
(both classification datasets plus the regression dataset), including a
live click of the Deployment tab's "Predict" button on both a
classification task (confirming the class-probability bar chart renders)
and the regression task (confirming a single numeric prediction renders) —
asserting no Streamlit exception banner appeared in either code path.
