# Prompts — Data Science Visual Foundations

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build the "teach beginner data science students" project from the source
repo's own prompt catalog: deep intuition and visual/interactive simulation
for Naive Bayes, model evaluation (confusion matrix, Type I/II errors,
ROC-AUC, cost matrix, precision/recall trade-off), gradient descent, and the
chain rule -> backpropagation, plus quizzes. Keep the same CRISP-DM tab
shell as the rest of the portfolio even though this project doesn't predict
a business KPI — reinterpret each phase pedagogically rather than skipping
the structure.
```

## Key follow-up decisions made along the way

* **A real dataset for Naive Bayes, not a toy one**: rather than a
  synthetic bag-of-words example, the real SMS Spam Collection dataset was
  sourced so the "type a message, see the math" demo classifies genuine
  text messages — the word-by-word log-probability breakdown reads directly
  from the trained `MultinomialNB.feature_log_prob_` array, so it's the
  actual model's arithmetic, not a re-derivation for display purposes.
* **Real data where it exists, a worked example where it doesn't**: gradient
  descent uses the real scikit-learn diabetes dataset (one real feature,
  BMI, against the real target) so the loss surface being descended is a
  genuine regression problem; the chain-rule/backprop section instead uses
  a fully worked *numeric* example (not a dataset) — the honest choice,
  since calculus intuition is better served by a small transparent example
  than by hiding the arithmetic inside a bigger real network.
* **CRISP-DM tabs kept even for a non-KPI project**: "Business
  Understanding" became "why these four ideas matter to every model in this
  portfolio," "Deployment" became the quiz — a deliberate choice to keep
  the whole portfolio's navigation muscle-memory consistent rather than
  giving this one project a bespoke structure.

## Verification

Headless Playwright drove all six tabs, then specifically exercised the
interactive elements: typing a spam-flavored message into the Naive Bayes
explainer (confirming a real word-level breakdown chart renders), dragging
the evaluation threshold slider, and submitting partial quiz answers via
the "Check my answers" button — confirming the scored/explained results
render with no Streamlit exception banner.
