# Prompts — Nano Transformer LLM

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build the "nano LLM" project from the source repo's own prompt catalog:
build a simple LLM with state-of-the-art primitives (real attention, real
transformer blocks) that fits on a laptop, same CRISP-DM tab structure as
the rest of the portfolio. Don't fake it with a pretrained model or a
Markov chain dressed up as a transformer — it should really train, live,
in the app, on real text, and really generate from what it learned.
```

## Key follow-up decisions made along the way

* **PyTorch was worth the one extra dependency**: the rest of the portfolio
  deliberately avoids heavy ML frameworks (see [project 06](../06_automl_model_tournament)'s
  AutoGluon-avoidance write-up), but a genuine attention mechanism needs
  real autograd to train live in a reasonable time budget. A quick
  benchmark confirmed the CPU-only PyTorch wheel installs in ~5 seconds and
  ~120MB — small enough to keep as an isolated, clearly-documented opt-in
  dependency (kept out of the shared root `requirements.txt`, installed
  separately per the README) rather than forcing it on every project in the
  portfolio.
* **Character-level over subword tokenization**: a from-scratch BPE
  tokenizer would add real complexity for negligible benefit at this scale
  (65 characters is already a tiny, fully-interpretable vocabulary) — a
  deliberate simplicity choice that also makes the Data Preparation tab's
  explanation much more concrete for a non-specialist reader.
* **Weekly-resampled seasonality precedent reused for train/val
  discipline**: the chronological (non-shuffled) 90/10 split mirrors the
  same "never shuffle time-ordered data" principle established in
  [project 05](../05_time_series_forecasting) — called out explicitly in
  the Data Preparation tab so the portfolio's consistency is visible to a
  reader, not just coincidental.
* **A real benchmark set the default hyperparameters**: before wiring the
  Streamlit UI, a standalone script timed 400 training steps at
  `n_embd=64, n_head=4, n_layer=3, block_size=64` (~9.4s, loss 4.2→2.1) —
  that measurement is what set the app's default `steps=600` slider value
  (a comfortable ~15s) rather than guessing at a "reasonable" training
  budget.
* **A real bug caught by browser verification**: an early version rendered
  a small char→id vocabulary preview as a transposed DataFrame with all
  column names blanked out to `""`, which Streamlit/PyArrow rejected with
  `ValueError: Duplicate column names found` — invisible until headless
  Playwright loaded that exact tab. Fixed by keeping the preview as a
  normal (non-transposed) two-column table instead.

## Verification

Headless Playwright drove all six tabs, with an extended wait specifically
on the Modeling tab (training genuinely takes several seconds — the
verification script waits up to 25s there rather than the ~2s used for
every other project's tabs) before asserting no exception banner, then
separately confirmed the Deployment tab's "Generate" button produces real,
non-empty Shakespeare-flavored text and the Evaluation tab's attention
heatmap renders a visible causal (lower-triangular) pattern.
