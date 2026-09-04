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

---

## 6 — Extending the portfolio: projects 05-08

After the first four projects were published, the follow-up instruction was
simple:

```text
please do the same for next 4 projects
```

Read in context of the whole session, "the same" meant: keep applying the
established process (pick technique-diverse real projects from the source
catalog, source genuinely real data for each, build a CRISP-DM Streamlit app
to the same bar, verify every tab headlessly, document honestly including
scope trade-offs, then commit and push) without re-litigating the count,
stack, or repo target already decided in step 2 — so the next four projects
were chosen and built directly rather than re-asking scope questions already
settled.

**Technique selection**, aimed at maximum diversity from what the first four
already covered (regression, clustering, association rules, imbalanced
classification):

```text
- 05 Time Series Forecasting: classical statistical forecasting (Holt-
  Winters, SARIMA) + an ML lag-feature approach, walk-forward backtested —
  a technique family entirely absent from projects 01-04.
- 06 AutoML Model Tournament: automated multi-algorithm, multi-hyperparameter
  search — the source repo's own AutoGluon project, scoped honestly onto
  plain scikit-learn (see that project's own PROMPTS.md for why).
- 07 Data Science Visual Foundations: the source repo's own "teach beginner
  data science students" project (Naive Bayes, model evaluation, gradient
  descent, chain rule/backprop, quizzes) — the one project in this portfolio
  that's a teaching tool, not a KPI predictor.
- 08 Nano Transformer LLM: the source repo's own "nano LLM" project — a
  real, from-scratch decoder-only Transformer, small enough to train live
  on a laptop CPU.
```

**New real datasets sourced** for this batch: the classic Box-Jenkins
Airline Passengers series and real Melbourne daily temperature readings (for
05); scikit-learn's own bundled real Breast Cancer / Wine / Diabetes
datasets, chosen specifically because they need no external download (for
06); the real UCI SMS Spam Collection (for 07); and Andrej Karpathy's real
Tiny Shakespeare corpus (for 08). See [`_build/fetch_extra_datasets.py`](./_build/fetch_extra_datasets.py).

**One new dependency, deliberately isolated**: project 08 needed real
PyTorch autograd for a genuine attention mechanism — the only project in the
portfolio with a dependency beyond the shared `requirements.txt`, installed
separately and documented as an explicit opt-in rather than forced on every
project (see that project's own `PROMPTS.md` for the benchmark that
justified it).

Every one of the four new apps was driven end-to-end with headless
Playwright before being considered done, exactly as projects 01-04 were —
this pass caught and fixed two real issues: a pandas/PyArrow
duplicate-column-name crash in project 08's vocabulary preview table, and
confirmed (rather than assumed) that project 05's SARIMA fits and project
08's live training both complete within a reasonable interactive time budget
before shipping default hyperparameters.

---

## 7 — Closing the portfolio: projects 09-11

After a status check-in on how many of the source repo's projects remained
unbuilt, the follow-up instruction was:

```text
lets work on the new concept ones. keep it simple as we did in earlier
projects
```

"The new concept ones" referred to three source-repo projects flagged in
that status check as genuinely new territory (as opposed to three others
that were near-duplicates of work already done — see that check-in's own
breakdown): a wide skills catalog, a full guided CRISP-DM curriculum, and a
portfolio-wide governance auditor. "Keep it simple" was read as a direct
instruction to prefer reusing existing datasets and existing patterns over
introducing new complexity — followed through concretely as **zero new
external downloads** across all three projects:

```text
- 09 Data Science Skills Lab: reuse the Mall Customers, Online Retail RFM,
  and Online Retail invoice data already built for projects 02 and 03 — a
  10-skill catalog (statistical testing, outlier detection, RFM scoring,
  cohort retention, PCA, business metrics) rather than the source repo's
  full 54-skill catalog.
- 10 CRISP-DM Master's Curriculum: Fisher's Iris dataset (1936), bundled
  directly in scikit-learn — a textbook-paced walkthrough of all six CRISP-DM
  phases with a concept check at each one, deliberately placing K-Means
  clustering next to a 3-classifier comparison on identical data so the
  unsupervised/supervised distinction is directly visible in one place.
- 11 Enterprise Data Science Audit: no external dataset at all — the
  portfolio's own source code (projects 01-10's real app.py and README.md
  files, read from disk at runtime) is the data.
```

**A real auditor, audited during its own construction**: project 11's first
working version was run against the (at-the-time 10-project) portfolio and
produced two visibly wrong results — an unfair leakage-risk penalty against
the two unsupervised projects (02, 03) for lacking a `train_test_split`
they have no use for, and a false-negative leakage flag against project 04
for a scaler that's actually fit safely on train-only data, just not via
the one `Pipeline(...)` idiom the first heuristic recognized. Both were
diagnosed by reading the flagged projects' actual code and confirming the
audit was wrong, then fixed in the auditor (a three-category rubric —
predictive / unsupervised / tool — and a second accepted safe-scaler
pattern) before the project was considered done, rather than shipping
results known to be unfair. A third, harder-to-fix gap (project 05's
hand-rolled NumPy metrics not matching any named-function keyword) was
identified but deliberately left as a disclosed limitation rather than
chased with more heuristics, in keeping with "keep it simple."

**Sequencing note**: projects 09 and 10's READMEs were written *after*
project 11's audit tool was first run against them — their absence at that
point showed up as real `Documentation: 0` scores in project 11's live
scorecard, which is disclosed directly in project 11's own README as
evidence the tool computes real results rather than a canned demo. Project
11's screenshots were then recaptured a final time after all documentation
was in place, so the shipped screenshots reflect the portfolio's completed
state rather than a mid-work snapshot.

As with every prior batch, all three new apps were driven end-to-end with
headless Playwright (including cycling through 5 of project 09's 10 skills
individually) before being considered done.
