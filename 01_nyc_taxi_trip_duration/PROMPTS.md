# Prompts — NYC Taxi Trip Duration Predictor

Part of the [Data Science Experiments](../README.md) portfolio. See the
[root prompt log](../PROMPTS.md) for the full session narrative.

## The ask

```text
Build a regression project predicting NYC taxi trip duration, following the
same CRISP-DM tab structure as the rest of the portfolio (Business
Understanding, Data Understanding, Data Preparation, Modeling, Evaluation,
Deployment). Use real NYC TLC data, not the deprecated Kaggle lat/lon CSV —
TLC's current schema uses zone IDs, so join the official zone shapefile for
approximate coordinates. Compare at least 2-3 regressors honestly, and make
the Deployment tab a real live trip estimator with an actual route map.
```

## Key follow-up decisions made along the way

* **Data source**: TLC's official Yellow Taxi parquet archive was streamed
  directly (via `fsspec` + `pyarrow`, footer/row-group reads) rather than
  downloading a full month locally — then sampled to 20,000 clean trips and
  feature-engineered into one compact CSV shipped in `data/`.
* **Coordinates**: the shapefile's projected CRS (`EPSG:2263`, NY State
  Plane feet) was reprojected to `EPSG:4326` (lat/lon) with `pyproj`, using
  lightweight `pyshp` + `shapely` instead of pulling in all of `geopandas`
  for a one-off centroid computation.
* **Map rendering**: Plotly 7 removed the old `scatter_mapbox`/`Scattermapbox`
  API in favor of `scatter_map`/`Scattermap` (MapLibre-based, no Mapbox
  token needed) — the app was updated to the new API after the first
  headless-browser check caught the `AttributeError`.
* **Chart theming**: a shared `theme.style_fig()` helper was added after
  noticing Plotly's `plotly_dark` template alone didn't darken the chart
  *paper* background inside Streamlit's own card styling — every chart now
  explicitly sets `paper_bgcolor`/`plot_bgcolor` to match the app's palette.

## Verification

Every tab was driven headlessly with Playwright (Chromium) after each
change — loading the app, clicking through all six tabs, and asserting no
Streamlit exception banner appeared — plus a live click-through of the
Deployment tab's "Estimate trip" button to confirm real predictions and the
route map render correctly end-to-end.
