"""NYC Taxi Trip Duration Predictor
==================================
A CRISP-DM-structured, end-to-end regression project: predict how long a
NYC yellow-taxi trip will take (and roughly what it will cost) from
pickup/dropoff zone, time of day, and trip characteristics.

Run:
    streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "data" / "nyc_taxi_trips.csv"

NUMERIC_FEATURES = [
    "passenger_count", "haversine_km", "pickup_hour", "pickup_weekday",
    "is_weekend", "is_rush_hour", "is_night",
]
CATEGORICAL_FEATURES = ["pickup_borough", "dropoff_borough"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "trip_duration_s"

MODEL_FACTORIES = {
    "Linear Regression (baseline)": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42
    ),
    "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(
        max_depth=8, learning_rate=0.08, max_iter=250, random_state=42
    ),
}


# --------------------------------------------------------------------------
# Data / model caching
# --------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])
    return df


@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    results = {}
    fitted = {}
    for name, factory in MODEL_FACTORIES.items():
        pipe = Pipeline([("pre", pre), ("model", factory())])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "MAE (s)": mean_absolute_error(y_test, pred),
            "RMSE (s)": root_mean_squared_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        }
        fitted[name] = pipe

    # feature importance from the tree model, mapped back through the encoder
    rf = fitted["Random Forest"]
    ohe: OneHotEncoder = rf.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = cat_names + NUMERIC_FEATURES
    importances = pd.Series(rf.named_steps["model"].feature_importances_, index=all_names)
    importances = importances.sort_values(ascending=False)

    return fitted, pd.DataFrame(results).T, importances, (X_test, y_test)


def predict_one(pipe, row: dict) -> float:
    x = pd.DataFrame([row])[FEATURES]
    return float(pipe.predict(x)[0])


def estimate_fare(duration_s: float, distance_km: float) -> float:
    # simple NYC-style fare heuristic for the deployment demo (not a trained model)
    base, per_km, per_min = 3.0, 1.75, 0.5
    return round(base + per_km * distance_km + per_min * (duration_s / 60), 2)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main():
    theme.apply("🚕", "NYC Taxi Trip Duration Predictor")
    theme.hero(
        "🚕", "NYC Taxi Trip Duration Predictor",
        "CRISP-DM regression pipeline on real NYC TLC Yellow Taxi trips (Jan 2023) — "
        "predicting trip duration from pickup/dropoff zone, time of day, and trip profile.",
        ["Regression", "scikit-learn", "Real TLC data · 20,000 trips", "Streamlit"],
    )

    df = load_data()

    with st.sidebar:
        st.subheader("📁 Dataset")
        st.metric("Trips", f"{len(df):,}")
        st.metric("Date range", f"{df['pickup_datetime'].dt.date.min()} → {df['pickup_datetime'].dt.date.max()}")
        st.caption(
            "Source: NYC TLC Yellow Taxi trip records (official public data, "
            "d37ci6vzurychx.cloudfront.net), joined with official taxi-zone "
            "centroids. See this project's README for the full pipeline."
        )

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding -------------------------------------------------
    with tabs[0]:
        st.subheader("Business objective")
        st.markdown(
            """
Riders want an **accurate ETA** before they commit to a trip; the platform wants a
**defensible fare estimate** and better **fleet allocation** signals. This mirrors the
original Kaggle *"NYC Taxi Trip Duration"* challenge, re-grounded in official TLC data.

**Success criteria (data science):** minimize MAE on held-out trip duration (seconds),
beating a naive baseline (mean duration per borough pair).

**Success criteria (business):** an estimator fast enough for a live product UI
(sub-100ms inference) and interpretable enough that ops can sanity-check surge behavior.
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Task</div>'
                     '<div class="ds-metric-value">Regression</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="ds-card"><div class="ds-metric-label">Target</div>'
                     '<div class="ds-metric-value">trip_duration_s</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Primary metric</div>'
                     '<div class="ds-metric-value">MAE (seconds)</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding ------------------------------------------------------
    with tabs[1]:
        st.subheader("Exploratory data analysis")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.histogram(df, x="trip_duration_s", nbins=60, title="Trip duration distribution (s)")
            theme.style_fig(fig, 340)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.histogram(df, x="haversine_km", nbins=60, title="Trip distance distribution (km, haversine)")
            theme.style_fig(fig, 340)
            st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns([1, 1])
        with c1:
            hourly = df.groupby("pickup_hour")["trip_duration_s"].mean().reset_index()
            fig = px.bar(hourly, x="pickup_hour", y="trip_duration_s",
                         title="Mean trip duration by pickup hour")
            theme.style_fig(fig, 340)
            st.plotly_chart(fig, width='stretch')
        with c2:
            borough_vol = df["pickup_borough"].value_counts().reset_index()
            borough_vol.columns = ["borough", "trips"]
            fig = px.pie(borough_vol, names="borough", values="trips", hole=0.5,
                         title="Trip volume by pickup borough")
            theme.style_fig(fig, 340)
            st.plotly_chart(fig, width='stretch')

        st.markdown("#### Pickup density map (sample of 4,000 trips)")
        sample = df.sample(min(4000, len(df)), random_state=1)
        fig = px.scatter_map(
            sample, lat="pickup_latitude", lon="pickup_longitude",
            color="pickup_borough", opacity=0.5, zoom=9.3, height=480,
            hover_data={"pickup_zone": True, "pickup_latitude": False, "pickup_longitude": False},
            map_style="dark",
        )
        theme.style_fig(fig, 480)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width='stretch')

        with st.expander("Correlation with trip duration"):
            corr = df[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
            corr = corr.sort_values()
            fig = px.bar(corr, orientation="h", title="Pearson correlation vs trip_duration_s")
            theme.style_fig(fig, 320)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

    # 3. Data Preparation ---------------------------------------------------------
    with tabs[2]:
        st.subheader("Feature engineering & cleaning")
        st.markdown(
            """
Starting from raw TLC parquet columns (`tpep_pickup_datetime`, `tpep_dropoff_datetime`,
`PULocationID`, `DOLocationID`, `trip_distance`, `fare_amount`, ...), the pipeline:

1. **Filters** implausible trips: duration outside `[60s, 3h]`, distance outside
   `[0.1, 50] km`, non-positive fares, passenger counts outside `[1, 6]`.
2. **Joins** official TLC taxi-zone centroids (reprojected from the shapefile,
   EPSG:2263 → EPSG:4326) onto pickup/dropoff `LocationID`s to recover approximate
   lat/lon (TLC replaced exact GPS with zone IDs from mid-2016 onward for privacy —
   this project follows the *current* official schema rather than the deprecated one).
3. **Derives** `haversine_km` (great-circle pickup→dropoff distance — independent of
   the meter-reported `trip_distance`, and available at *prediction* time from just
   two zones), plus calendar features: `pickup_hour`, `pickup_weekday`, `is_weekend`,
   `is_rush_hour` (7-9am / 4-7pm), `is_night` (10pm-5am).
4. **One-hot encodes** pickup/dropoff borough inside a `ColumnTransformer` fit
   *only* on the training split (zero leakage into the held-out test set).
            """
        )
        before, after = 40 + len(df) // 20, len(df)  # illustrative note, real filter ratio documented in README
        st.info(
            "Full cleaning ratios (raw TLC month → final asset) are documented in "
            "`_build/build_nyc_taxi_dataset.py` referenced from the project README — "
            "this app ships the already-cleaned, feature-engineered CSV so it starts instantly.",
            icon="🧹",
        )
        st.dataframe(df[FEATURES + [TARGET]].head(15), width='stretch')

    # 4. Modeling -------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Model comparison")
        st.caption("Three models trained on an 80/20 split, identical feature set, identical encoder fit on train only.")
        fitted, results, importances, (X_test, y_test) = train_models(df)

        def _highlight_best(row):
            is_best = row.name == results["MAE (s)"].idxmin()
            style = "background-color:#134e2e;color:#6ee7f2;font-weight:600" if is_best else ""
            return [style] * len(row)

        st.dataframe(
            results.style.format({"MAE (s)": "{:.1f}", "RMSE (s)": "{:.1f}", "R2": "{:.3f}"})
            .apply(_highlight_best, axis=1),
            width='stretch',
        )
        st.caption("🟢 highlighted row = best model on held-out MAE (lower is better).")

        st.markdown("#### Feature importance — Random Forest")
        fig = px.bar(importances.head(15)[::-1], orientation="h")
        theme.style_fig(fig, 420)
        fig.update_layout(showlegend=False, xaxis_title="Gini importance", yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    # 5. Evaluation -------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Held-out evaluation")
        fitted, results, importances, (X_test, y_test) = train_models(df)
        best_name = results["MAE (s)"].idxmin()
        st.success(f"Best model on held-out MAE: **{best_name}** "
                   f"({results.loc[best_name, 'MAE (s)']:.1f}s MAE, R²={results.loc[best_name, 'R2']:.3f})",
                   icon="🏆")

        best_pipe = fitted[best_name]
        pred = best_pipe.predict(X_test)
        resid = y_test.values - pred

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(x=y_test, y=pred, opacity=0.35, labels={"x": "Actual duration (s)", "y": "Predicted duration (s)"},
                              title=f"Predicted vs actual — {best_name}")
            lims = [0, max(y_test.max(), pred.max())]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#fb7185", dash="dash"), name="perfect"))
            theme.style_fig(fig, 380)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.histogram(resid, nbins=60, title="Residual distribution (actual - predicted)")
            theme.style_fig(fig, 380)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        err_df = X_test.copy()
        err_df["abs_err"] = np.abs(resid)
        by_hour = err_df.groupby("pickup_hour")["abs_err"].mean().reset_index()
        fig = px.line(by_hour, x="pickup_hour", y="abs_err", markers=True,
                      title="Mean absolute error by pickup hour (where does the model struggle?)")
        theme.style_fig(fig, 340)
        st.plotly_chart(fig, width='stretch')

    # 6. Deployment ----------------------------------------------------------------
    with tabs[5]:
        st.subheader("Live trip estimator")
        fitted, results, importances, _ = train_models(df)
        best_name = results["MAE (s)"].idxmin()
        model_choice = st.selectbox("Model", list(fitted.keys()), index=list(fitted.keys()).index(best_name))
        pipe = fitted[model_choice]

        boroughs = sorted(df["pickup_borough"].dropna().unique())
        zone_lookup = df.drop_duplicates("pickup_zone")[["pickup_zone", "pickup_borough", "pickup_latitude", "pickup_longitude"]]
        zone_lookup_do = df.drop_duplicates("dropoff_zone")[["dropoff_zone", "dropoff_borough", "dropoff_latitude", "dropoff_longitude"]]

        c1, c2, c3 = st.columns(3)
        with c1:
            pu_borough = st.selectbox("Pickup borough", boroughs, index=boroughs.index("Manhattan") if "Manhattan" in boroughs else 0)
            pu_options = zone_lookup[zone_lookup["pickup_borough"] == pu_borough]["pickup_zone"].sort_values().unique()
            pu_zone = st.selectbox("Pickup zone", pu_options)
        with c2:
            do_borough = st.selectbox("Dropoff borough", boroughs, index=0)
            do_options = zone_lookup_do[zone_lookup_do["dropoff_borough"] == do_borough]["dropoff_zone"].sort_values().unique()
            do_zone = st.selectbox("Dropoff zone", do_options)
        with c3:
            pickup_hour = st.slider("Pickup hour", 0, 23, 18)
            weekday = st.selectbox("Day of week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=4)
            passengers = st.slider("Passengers", 1, 6, 1)

        pu_row = zone_lookup[zone_lookup["pickup_zone"] == pu_zone].iloc[0]
        do_row = zone_lookup_do[zone_lookup_do["dropoff_zone"] == do_zone].iloc[0]

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            return 2 * R * np.arcsin(np.sqrt(a))

        dist_km = haversine(pu_row["pickup_latitude"], pu_row["pickup_longitude"],
                             do_row["dropoff_latitude"], do_row["dropoff_longitude"])
        weekday_idx = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(weekday)

        row = {
            "passenger_count": passengers,
            "haversine_km": dist_km,
            "pickup_hour": pickup_hour,
            "pickup_weekday": weekday_idx,
            "is_weekend": int(weekday_idx >= 5),
            "is_rush_hour": int(pickup_hour in [7, 8, 9, 16, 17, 18, 19]),
            "is_night": int(pickup_hour >= 22 or pickup_hour <= 5),
            "pickup_borough": pu_borough,
            "dropoff_borough": do_borough,
        }

        if st.button("🚕 Estimate trip", type="primary"):
            duration_s = predict_one(pipe, row)
            fare = estimate_fare(duration_s, dist_km)
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated duration", f"{duration_s/60:.1f} min")
            m2.metric("Great-circle distance", f"{dist_km:.2f} km")
            m3.metric("Estimated fare", f"${fare:.2f}")

            route = pd.DataFrame({
                "lat": [pu_row["pickup_latitude"], do_row["dropoff_latitude"]],
                "lon": [pu_row["pickup_longitude"], do_row["dropoff_longitude"]],
                "label": [f"Pickup: {pu_zone}", f"Dropoff: {do_zone}"],
            })
            fig = go.Figure(go.Scattermap(
                lat=route["lat"], lon=route["lon"], mode="markers+lines+text",
                text=route["label"], textposition="top right",
                marker=dict(size=14, color=["#6ee7f2", "#fb7185"]),
                line=dict(width=3, color="#a78bfa"),
            ))
            theme.style_fig(fig, 460)
            fig.update_layout(
                map=dict(style="dark", center=dict(lat=route["lat"].mean(), lon=route["lon"].mean()), zoom=10.5),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("Pick a route and click **Estimate trip**.")

    theme.footer("01 · NYC Taxi Trip Duration Predictor")


if __name__ == "__main__":
    main()
