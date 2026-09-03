# Prompts — Customer Segmentation & Clustering

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build an unsupervised clustering project using a popular Kaggle-style
dataset, same CRISP-DM tab structure as the rest of the portfolio. Use your
creativity — don't just do the textbook Mall Customers demo, add a second,
business-grounded angle (RFM behavioral segmentation) if you can source real
data for it. Compare at least K-Means against one other algorithm, show an
elbow/silhouette analysis to justify k, and make the Deployment tab actually
assign a new hypothetical customer to a segment.
```

## Key follow-up decisions made along the way

* **Second dataset added for creative depth**: rather than stop at the
  200-row Mall Customers demo, the UCI "Online Retail" invoice data (already
  being downloaded for the market-basket-mining project) was reused to
  derive a proper RFM table — letting one app demonstrate both classic
  demographic segmentation and real behavioral segmentation, switchable live.
* **Log-transform for RFM**: an early run showed the RFM dataset's
  silhouette-optimal `k` collapsing to 2 with visibly outlier-dominated
  clusters. Standard RFM practice — `log1p` on `frequency` and `monetary`
  before scaling — was added (dataset-conditionally, via a `log_features`
  config key), with the reasoning surfaced directly in the Data Preparation
  tab rather than silently applied.
* **DBSCAN/Agglomerative honesty**: rather than fake a "predict" for
  non-centroid-based algorithms in the Deployment tab, the app explicitly
  tells the user to switch to K-Means for live scoring — an accurate
  statement about those algorithms' limitations instead of a hidden hack.

## Verification

Both datasets were exercised end-to-end with headless Playwright: switching
the sidebar radio, re-running the elbow/silhouette sweep, fitting K-Means,
and clicking "Assign segment" in the Deployment tab — confirming no
Streamlit exception banner appeared in either dataset's code path (this is
how a `KeyError` from an empty-DataFrame edge case in a *different* project,
[03](../03_market_basket_mining/PROMPTS.md), was originally caught — the
same click-through discipline was applied here even though this app didn't
hit that particular bug).
