# Prompts — Time Series Forecasting

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build a time series forecasting project on real, public-domain data, same
CRISP-DM tab structure as the rest of the portfolio. Compare classical
statistical forecasters (exponential smoothing, SARIMA) against a
naive baseline and an ML approach. Evaluate with walk-forward backtesting,
never a random split — that's the one mistake this project has to get
visibly right. Make the Deployment tab a real live forecast with an
uncertainty band.
```

## Key follow-up decisions made along the way

* **Two real datasets, different seasonal regimes**: the classic monthly
  Airline Passengers series (strong trend + yearly seasonality, 144 points)
  was paired with real daily Melbourne temperature readings, deliberately
  chosen to stress-test the pipeline with a *much* longer, noisier series.
* **Resampling the daily series to weekly**: an early plan to run SARIMA
  directly on 3,650 daily points with `m=365` seasonality was dropped once
  it became clear that seasonal order is computationally prohibitive for an
  interactive demo (state-space dimensionality scales with the seasonal
  period). Resampling to weekly means (`m=52`) was chosen as an honest,
  documented trade-off — the underlying readings stay real and unmodified,
  only the aggregation changes — rather than silently downsampling and
  calling it "daily."
* **Recursive multi-step ML forecasting**: the gradient-boosting forecaster
  is trained once on lag features, then generates a whole horizon by
  feeding each prediction back in as the next step's lag — the only honest
  way to multi-step forecast with a model that only ever learned one-step-
  ahead, and exactly what a production system would have to do.
* **`macOS` shell note**: `timeout` isn't available by default on macOS —
  an early verification command failed with `command not found: timeout`
  before switching to plain background (`&`) + `sleep` for headless
  Streamlit runs.

## Verification

Both datasets were driven through all six tabs with headless Playwright,
including the full 3-fold walk-forward backtest (SARIMA fitting is the slow
step — the weekly-temperature run was allowed up to 45s to complete before
asserting no exception banner appeared), plus a live click of the
Deployment tab's "Forecast" button confirming the forecast fan and
uncertainty band render correctly end-to-end.
