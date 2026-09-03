# Prompts — Market Basket / Associative Pattern Mining

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build an associative pattern mining project on a popular real dataset, same
CRISP-DM tab structure as the rest of the portfolio. Compare Apriori and
FP-Growth, surface support/confidence/lift properly, and make the Deployment
tab a genuinely usable "frequently bought together" recommender — not just a
static rules table.
```

## Key follow-up decisions made along the way

* **Dataset choice**: the UCI "Online Retail" dataset (already sourced for
  this project) doubles as the source for project 02's RFM angle — one real
  download, two projects, rather than inventing a separate synthetic
  transactions dataset.
* **A real bug caught by browser verification**: the first version of the
  Deployment-tab recommender computed `matches = matches[~matches["consequents"]...]`
  as a *second* boolean-mask filter chained onto an already-empty-result
  DataFrame. Pandas silently collapsed that empty frame to **zero columns**
  (not just zero rows) in this edge case, so a later `.sort_values("lift")`
  raised `KeyError: 'lift'` — invisible in casual testing (empty carts are
  easy to skip past by hand) but caught immediately by the headless
  Playwright click-through, which exercises the default pre-filled cart on
  every run. Fixed by computing both boolean masks against the *original*
  `rules` frame and combining them with `&` in one filter, instead of
  chaining two separate frame subsets.
* **Recommendation transparency**: recommendations show *which* cart item(s)
  triggered each suggestion ("because you have..."), so the output is
  auditable against the actual mined rule rather than a black box.

## Verification

Headless Playwright drove all six tabs (asserting no exception banner),
then specifically exercised the Deployment tab: opening the product
multiselect, adding a second real product to the cart, and confirming both
the recommendation table and the cart→recommendation graph render with real
data (this is the run that caught the `KeyError` above).
