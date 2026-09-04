# Prompts — CRISP-DM Master's Curriculum

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build the source repo's "CRISP-DM Master's Platform" project — a textbook-
quality guided walkthrough of every CRISP-DM phase on one dataset, covering
EDA, clustering, classification, and a synthesis conclusion, with concept
checks. Keep it simple: don't duplicate the heavy modeling already done in
projects 01-06 — pick a dataset and scope where the *teaching structure*
is the point, not another deep model comparison.
```

## Key follow-up decisions made along the way

* **Iris over a new/heavier dataset**: rather than source new data or reuse
  one of the already-deep business datasets (which would invite unhelpful
  comparison with projects 01-06's fuller model tournaments), Fisher's
  1936 Iris dataset was chosen specifically for its teaching pedigree —
  zero download (bundled in scikit-learn), small enough to move through six
  chapters at a readable pace, and famous enough that its own history *is*
  part of the business-understanding lesson.
* **Clustering next to classification, deliberately**: the single biggest
  design decision was running K-Means and 3 classifiers on the *same* split
  of the *same* data in the *same* tab, specifically so a reader can see
  the unsupervised/supervised distinction directly (Adjusted Rand Index vs.
  accuracy, side-by-side scatter plots) rather than having to remember and
  compare two entirely separate projects.
* **Concept checks per chapter, not one quiz at the end**: unlike
  [project 07](../07_data_science_visual_foundations)'s closing quiz, this
  project's checks are woven into each chapter immediately after the
  relevant content, with a running score in the sidebar — reinforcing one
  idea at a time rather than testing recall of the whole journey at once.

## Verification

Headless Playwright drove all six tabs, then specifically: answered a
concept-check radio button on Chapter 1 and confirmed the sidebar's
"Curriculum progress" counter updated; and clicked the Chapter 6 "Identify
species" button, confirming a real prediction and class-probability chart
render using the models fitted earlier in the same session via
`st.session_state` — no Streamlit exception banner in either case.
