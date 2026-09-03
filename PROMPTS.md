# 📜 Prompt Log

This document records, in order, the prompts and directives that produced
this repository — following the same "verbatim reproduction catalog" idea as
the source assignment repo's own `PROMPTS.md`, but for this session. Each
project folder also has its own short `PROMPTS.md` with the project-specific
ask.

---

## 1 — The assignment

The starting brief (paraphrased from the course assignment page):

```text
Replicate the data science experiments [the instructor] did in your favorite
coding assistant - I provided the prompts in the repository
https://github.com/dlmastery/data_science_examples

Make sure you publish them in a github repo and walk through these in a
YouTube video and upload and link to README.md

Note: use your creativity and not need to do exactly like above - improvise
if possible. The minimum bar is above.

Publish a link to a YouTube video which walks through the code and UX of
each of the projects thoroughly in README.md. I provided PROMPTS.md with
what prompts I used. Use your creativity to do more in each app and explain.
```

And the direct instruction to the assistant (Claude Code) that kicked off
the work in this session:

```text
Above is the assignment description. I want you to go through it. Clone the
repo in this folder and then start working on it.

Your goal is to complete this assignment. You can leave the video generation
and anything related to video. But other than that, please work on it and
complete it in this folder.
```

## 2 — Scoping the work

The source repository turned out to contain **16 separate full-stack
(FastAPI + React) applications** — an enterprise-scale portfolio, not
realistically reproducible at full depth in one session. Rather than build
16 shallow stubs, the assistant proposed (and the user confirmed) a smaller,
deeper scope:

```text
How many of the 16 project ideas should actually be built (fully working,
not stubs)? → 4 projects, chosen for technique diversity (regression,
clustering, association-rule mining, anomaly detection).

What tech stack/UX depth for each project's "app"? → Streamlit single-app
per project instead of a separate FastAPI backend + React frontend — still
genuinely interactive (live widgets, live charts, live inference), a
fraction of the moving parts.

Where should the repo be published? → a new public GitHub repo under the
authenticated account.
```

## 3 — Sourcing real data

Every project's dataset was sourced live from a public, well-known source
rather than fabricated — a deliberate improvement on doing this quickly with
synthetic stand-ins:

```text
- NYC Taxi: stream NYC TLC's own official Yellow Taxi parquet archive
  directly (d37ci6vzurychx.cloudfront.net), join the official taxi-zone
  shapefile (reprojected EPSG:2263 → EPSG:4326) for approximate pickup/
  dropoff coordinates, since TLC deprecated raw lat/lon for privacy in 2016.
- Customer segmentation: the classic Kaggle "Mall Customers" panel, plus a
  second real angle — RFM aggregates derived from the UCI "Online Retail"
  transactional dataset.
- Market basket mining: the same UCI "Online Retail" dataset, restricted to
  UK invoices and the 120 most frequent SKUs, one-hot encoded into a basket
  matrix.
- Fraud detection: the Kaggle/ULB "Credit Card Fraud Detection" dataset,
  stratified down to a repo-friendly size while keeping every one of the
  492 real confirmed frauds.
```

See [`_build/`](./_build) for the exact scripts, and each project's own
README for dataset-specific detail.

## 4 — Building each app to a consistent bar

For every project, the same instruction was effectively applied: build a
CRISP-DM-structured Streamlit app (Business Understanding → Data
Understanding → Data Preparation → Modeling → Evaluation → Deployment tabs),
compare multiple real models/algorithms rather than one, evaluate with the
metric that actually matters for that problem (not just accuracy), and end
with a genuinely interactive "Deployment" tab a non-technical reviewer could
use without reading code.

```text
Build the [taxi duration / customer segmentation / market basket / fraud]
project as a self-contained Streamlit app using the shared common/theme.py
design system. Compare at least 2-3 models honestly. Use the metric that's
actually right for this problem. Make the Deployment tab a real, working
predictor — not a mockup.
```

Every app was then run headlessly and driven with Playwright (Chromium) to
catch real runtime exceptions before they shipped, and to capture the
screenshots embedded in each project's README — the same "browser-test
everything, not just the final inference" discipline the source repo's own
prompt log calls out.

## 5 — Documentation & publishing

```text
Write each project's own README (business framing, CRISP-DM write-up,
architecture notes, a full screenshot tour) and its own PROMPTS.md. Write a
top-level README tying the whole portfolio together, with a placeholder for
the YouTube walkthrough link. Initialize git, create a public GitHub repo,
and push everything.
```

(Video recording itself was explicitly out of scope for this session, per
the assignment's own note that it's fine to defer — the README link is left
as a placeholder for the user to fill in after recording.)
