# Prompts — Enterprise Data Science Audit

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build the source repo's "Enterprise Data Science Audit" project — but make
it a real auditor, not a hardcoded report. It should genuinely inspect this
portfolio's own code and produce real findings, the same CRISP-DM tab
structure as the rest of the portfolio, kept simple: plain regex/substring
checks over real files, no new dependencies, no external data (the
portfolio itself is the dataset).
```

## Key follow-up decisions made along the way

* **Genuinely live, not a snapshot**: the very first design decision was
  that this app must `Path.read_text()` every other project's real
  `app.py`/`README.md` at runtime and compute every score from that content
  — never a hardcoded results table pretending to be an audit. This was
  verified by watching the numbers change as documentation gaps in
  [09](../09_data_science_skills_lab) and [10](../10_crispdm_masters_curriculum)
  (their READMEs not yet written at that point) showed up as real
  `Documentation: 0` scores in the live scorecard, before those READMEs
  were written — the tool caught a true, then-real gap in this very
  session, which is the strongest possible evidence it isn't scripted.
* **Fixing the auditor's own bugs before shipping it**: running the first
  version surfaced two categorization/heuristic mistakes (detailed in this
  project's own README's "What the audit actually found" section) — an
  unfair penalty against unsupervised projects for lacking a
  `train_test_split`, and a false negative against project 04's genuinely
  safe (but non-`Pipeline`-idiom) scaler fit. Both were real bugs in the
  *auditor*, caught by looking hard at results that didn't match what a
  careful human review of those projects would conclude, and fixed rather
  than shipped — treating the audit tool with the same rigor it applies to
  everything else.
* **Three project categories instead of one flat rubric**: rather than
  chase every possible false positive with more special-casing (scope
  creep), the fix that generalizes was to formally split projects into
  Predictive / Unsupervised / Tool categories and apply only the dimensions
  that are actually meaningful for each — documented directly in-app so
  the judgment call is visible, not hidden inside scoring code.
* **A deliberate stopping point**: after those two fixes, a third,
  genuinely harder-to-fix gap surfaced (project 05's hand-rolled NumPy
  metrics vs. named sklearn functions) — rather than keep patching
  heuristics indefinitely (in tension with "keep it simple"), this was
  documented as an honest, disclosed limitation instead of chased to zero.

## Verification

Headless Playwright drove all six tabs against the live repository,
confirmed the scorecard table renders real (non-identical, non-hardcoded)
per-project numbers, exercised the Evaluation tab's project-selector radar
chart across a specific project, and submitted a real code snippet through
the Deployment tab's "audit your own code" flow — confirming the generic
checks run correctly on arbitrary pasted code, not just this repo's own
files.
