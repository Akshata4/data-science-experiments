# Prompts — Data Science Skills Lab

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build the "data science skills lab" project from the source repo's own
catalog — a wide catalog of individual analytical skills, each demonstrated
on real data, same CRISP-DM tab structure as the rest of the portfolio.
Keep it simple: don't chase the source repo's full 54-skill catalog or
source new datasets — reuse data already built for earlier projects.
```

## Key follow-up decisions made along the way

* **Zero new data sourcing, by design**: every one of the 10 skills runs on
  CSVs already built for [project 02](../02_customer_segmentation_clustering)
  and [project 03](../03_market_basket_mining) — a deliberate "keep it
  simple" choice per explicit instruction, rather than adding a fourth
  dataset just for variety.
* **10 curated skills over 54 exhaustive ones**: chosen to span every
  category a real toolkit needs (statistics, data quality, data prep,
  segmentation, modeling, business translation) rather than maximizing
  count — quality and breadth of *category* over raw quantity.
* **A registry pattern, not 10 copy-pasted tab bodies**: each skill is one
  function in a `SKILLS` dict (`name -> {fn, category, dataset}`), so the
  Modeling tab's "run a skill" selector, the Data Preparation tab's catalog
  grid, and the Evaluation tab's category breakdown all derive from the
  same single source of truth instead of three places to keep in sync.

## Verification

Headless Playwright drove all six tabs, then specifically cycled the
Modeling tab's skill selector through 5 of the more complex skills (RFM
scoring, cohort retention, PCA, chi-square, business metrics) — each
confirmed to render with no Streamlit exception banner. This caught a
Playwright *test-script* ambiguity (the skill-selector dropdown option text
also appears in the Data Preparation tab's always-rendered catalog grid,
so `get_by_text(..., exact=True)` matched two elements) rather than an app
bug — fixed by scoping the click to the dropdown's own listbox element.
