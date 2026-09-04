"""Time Series Forecasting
=============================
A CRISP-DM forecasting project comparing four real forecasting approaches —
Naive, Seasonal Naive, Holt-Winters exponential smoothing, SARIMA, and a
lag-feature gradient-boosting regressor — on two classic, real, public-domain
time series, evaluated with **walk-forward backtesting** (never a random
train/test split, which would leak future values into training for a
time-ordered problem).

Run:
    streamlit run app.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS = {
    "Airline Passengers (monthly, 1949-1960)": {
        "path": DATA_DIR / "airline_passengers.csv", "date_col": "Month", "value_col": "Passengers",
        "freq": "MS", "season_length": 12,
        "source": "Classic Box-Jenkins airline dataset: monthly international airline "
                   "passenger totals, Jan 1949-Dec 1960 (144 points). Strong trend and "
                   "multiplicative yearly seasonality.",
        "resample": None,
    },
    "Melbourne Daily Min Temperature (resampled weekly)": {
        "path": DATA_DIR / "daily_min_temperatures.csv", "date_col": "Date", "value_col": "Temp",
        "freq": "W", "season_length": 52,
        "source": "Daily minimum temperatures, Melbourne, Australia, 1981-1990 (3,650 daily "
                   "readings). Resampled to weekly means here (52-week seasonality) so SARIMA "
                   "stays tractable in a browser demo; the underlying readings are real, unmodified.",
        "resample": "W",
    },
}


@st.cache_data
def load_series(name: str) -> pd.Series:
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["path"])
    df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]])
    s = df.set_index(cfg["date_col"])[cfg["value_col"]].asfreq(
        "MS" if cfg["resample"] is None else "D"
    )
    s = s.interpolate()
    if cfg["resample"]:
        s = s.resample(cfg["resample"]).mean()
    s.index.freq = cfg["freq"]
    return s


# ---------------------------------------------------------------------------
# Forecasters — each takes a training Series and horizon h, returns h forecasts
# ---------------------------------------------------------------------------
def naive_forecast(train: pd.Series, h: int, **_) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def seasonal_naive_forecast(train: pd.Series, h: int, season_length: int, **_) -> np.ndarray:
    last_season = train.iloc[-season_length:].values
    reps = int(np.ceil(h / season_length))
    return np.tile(last_season, reps)[:h]


def holt_winters_forecast(train: pd.Series, h: int, season_length: int, **_) -> np.ndarray:
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=season_length,
        initialization_method="estimated",
    ).fit(optimized=True)
    return model.forecast(h).values


def sarima_forecast(train: pd.Series, h: int, season_length: int, **_) -> np.ndarray:
    model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 0, season_length),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    return model.forecast(h).values


def _make_lag_features(s: pd.Series, season_length: int) -> pd.DataFrame:
    lags = sorted(set([1, 2, 3, season_length]))
    df = pd.DataFrame({"y": s})
    for lag in lags:
        df[f"lag_{lag}"] = s.shift(lag)
    df["rolling_mean_3"] = s.shift(1).rolling(3).mean()
    df["t"] = np.arange(len(s))
    return df.dropna()


def ml_forecast(train: pd.Series, h: int, season_length: int, **_) -> np.ndarray:
    feat_df = _make_lag_features(train, season_length)
    X, y = feat_df.drop(columns="y"), feat_df["y"]
    model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
    model.fit(X, y)

    history = list(train.values)
    preds = []
    lags = sorted(set([1, 2, 3, season_length]))
    for step in range(h):
        row = {f"lag_{lag}": history[-lag] for lag in lags}
        row["rolling_mean_3"] = np.mean(history[-3:])
        row["t"] = len(history)
        x = pd.DataFrame([row])[X.columns]
        pred = float(model.predict(x)[0])
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


MODELS = {
    "Naive (last value)": naive_forecast,
    "Seasonal Naive": seasonal_naive_forecast,
    "Holt-Winters": holt_winters_forecast,
    "SARIMA": sarima_forecast,
    "Gradient Boosting (lag features)": ml_forecast,
}


def mape(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.mean(np.abs((actual - pred) / np.where(actual == 0, 1e-6, actual))) * 100)


@st.cache_data(show_spinner=True)
def walk_forward_backtest(series_name: str, h: int, n_folds: int = 3):
    series = load_series(series_name)
    season_length = DATASETS[series_name]["season_length"]
    min_train = max(season_length * 3, 24)

    fold_starts = np.linspace(min_train, len(series) - h, n_folds).astype(int)
    rows = []
    last_fold_detail = None
    for fold_i, cut in enumerate(fold_starts):
        train, test = series.iloc[:cut], series.iloc[cut:cut + h]
        if len(test) < h:
            continue
        fold_detail = {"train_end": series.index[cut - 1], "actual": test}
        for name, fn in MODELS.items():
            try:
                pred = fn(train, h, season_length=season_length)
            except Exception:
                pred = np.repeat(train.iloc[-1], h)
            mae = float(np.mean(np.abs(test.values - pred)))
            rmse = float(np.sqrt(np.mean((test.values - pred) ** 2)))
            rows.append({"fold": fold_i, "model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape(test.values, pred)})
            fold_detail[name] = pred
        last_fold_detail = fold_detail

    results = pd.DataFrame(rows)
    summary = results.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().sort_values("MAE")
    return summary, last_fold_detail, season_length


def main():
    theme.apply("📈", "Time Series Forecasting")
    theme.hero(
        "📈", "Time Series Forecasting",
        "CRISP-DM forecasting pipeline comparing Naive, Seasonal Naive, Holt-Winters, "
        "SARIMA, and gradient-boosted lag features — validated with walk-forward "
        "backtesting on two real, public-domain time series.",
        ["Forecasting", "statsmodels · SARIMA", "Walk-forward backtesting", "Streamlit"],
    )

    with st.sidebar:
        st.subheader("📁 Dataset")
        dataset_name = st.radio("Choose a series", list(DATASETS.keys()))
        cfg = DATASETS[dataset_name]
        series = load_series(dataset_name)
        st.metric("Observations", f"{len(series):,}")
        st.metric("Season length", cfg["season_length"])
        st.caption(cfg["source"])

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Business objective")
        st.markdown(
            """
Operations teams need forecasts to plan capacity, staffing, and inventory —
and they need **honest uncertainty**, not just a point estimate. Time series
forecasting is also a classic place for a subtle but serious data science
mistake: **evaluating with a random train/test split**. A random split lets
the model "see" data from *after* the point it's supposed to be predicting,
producing wildly over-optimistic offline metrics that collapse in
production. Every evaluation in this app instead uses **walk-forward
backtesting** — training only on the past, forecasting strictly forward.

**Primary metric:** MAE and MAPE, averaged across multiple rolling-origin
folds (not a single lucky/unlucky split).
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Task</div>'
                     '<div class="ds-metric-value">Forecasting</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="ds-card"><div class="ds-metric-label">Validation</div>'
                     '<div class="ds-metric-value">Walk-forward</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Primary metric</div>'
                     '<div class="ds-metric-value">MAE / MAPE</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding --------------------------------------------------------
    with tabs[1]:
        st.subheader("Exploratory data analysis")
        fig = px.line(series, title="Full series")
        theme.style_fig(fig, 340)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

        decomp = seasonal_decompose(series, model="additive", period=cfg["season_length"])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(decomp.trend, title="Trend component")
            theme.style_fig(fig, 300); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.line(decomp.seasonal.iloc[:cfg["season_length"] * 2], title="Seasonal component (first 2 cycles)")
            theme.style_fig(fig, 300); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        nlags = min(40, len(series) // 2 - 1)
        acf_vals = acf(series.dropna(), nlags=nlags)
        pacf_vals = pacf(series.dropna(), nlags=nlags)
        with c1:
            fig = px.bar(x=list(range(len(acf_vals))), y=acf_vals, title="Autocorrelation (ACF)")
            theme.style_fig(fig, 300); fig.update_layout(showlegend=False, xaxis_title="lag", yaxis_title="ACF")
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.bar(x=list(range(len(pacf_vals))), y=pacf_vals, title="Partial autocorrelation (PACF)")
            theme.style_fig(fig, 300); fig.update_layout(showlegend=False, xaxis_title="lag", yaxis_title="PACF")
            st.plotly_chart(fig, width='stretch')

    # 3. Data Preparation ------------------------------------------------------------
    with tabs[2]:
        st.subheader("Split strategy & feature engineering")
        st.markdown(
            f"""
1. **Time-ordered split only** — every evaluation below trains on
   observations strictly before a cutoff and forecasts strictly after it.
   No shuffling, ever.
2. **Walk-forward backtesting** — the cutoff is advanced across **3 rolling
   origins**, so each model is scored on 3 independent forecast windows, not
   one. This is the time series equivalent of k-fold cross-validation.
3. **Lag features for the ML model** — the gradient boosting forecaster
   never sees a raw timestamp as a magic feature; it sees `lag_1`, `lag_2`,
   `lag_3`, `lag_{cfg['season_length']}` (same period last cycle), and a
   3-period rolling mean. Multi-step forecasts are produced **recursively**
   — each predicted value becomes an input lag for the next step, exactly
   as it would need to work in production.
            """
        )
        feat_preview = _make_lag_features(series, cfg["season_length"]).head(8)
        st.dataframe(feat_preview, width='stretch')

    # 4. Modeling -----------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Model tournament — walk-forward backtest")
        h = st.slider("Forecast horizon per fold (periods)", 6, 24, 12)
        summary, last_fold, season_length = walk_forward_backtest(dataset_name, h)
        st.caption(f"Averaged over 3 rolling-origin folds, {h}-period horizon each.")
        st.dataframe(
            summary.style.format("{:.2f}").background_gradient(subset=["MAE"], cmap="Greens_r"),
            width='stretch',
        )

        st.markdown("#### Most recent fold — forecast vs actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=last_fold["actual"].index, y=last_fold["actual"].values,
                                  mode="lines+markers", name="Actual", line=dict(color="#e6ebf5", width=3)))
        colors = ["#6ee7f2", "#a78bfa", "#fbbf24", "#fb7185", "#34d399"]
        for (name, _), color in zip(MODELS.items(), colors):
            fig.add_trace(go.Scatter(x=last_fold["actual"].index, y=last_fold[name],
                                      mode="lines", name=name, line=dict(color=color, dash="dot")))
        theme.style_fig(fig, 440)
        st.plotly_chart(fig, width='stretch')

    # 5. Evaluation -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Model comparison")
        summary, last_fold, season_length = walk_forward_backtest(dataset_name, 12)
        best = summary["MAE"].idxmin()
        st.success(f"Best model on average walk-forward MAE: **{best}**", icon="🏆")

        fig = px.bar(summary.reset_index(), x="model", y="MAE", color="model", title="Average MAE by model (lower is better)")
        theme.style_fig(fig, 380)
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, width='stretch')

        resid = last_fold["actual"].values - last_fold[best]
        fig = px.histogram(resid, nbins=15, title=f"Residual distribution — {best} (most recent fold)")
        theme.style_fig(fig, 300); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # 6. Deployment -----------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Generate a live forecast")
        c1, c2 = st.columns(2)
        model_name = c1.selectbox("Model", list(MODELS.keys()), index=2)
        horizon = c2.slider("Forecast how many periods ahead?", 4, 36, 12)

        if st.button("📈 Forecast", type="primary"):
            fn = MODELS[model_name]
            point = fn(series, horizon, season_length=cfg["season_length"])

            # naive residual-based interval from the walk-forward backtest of this model
            summary, _, _ = walk_forward_backtest(dataset_name, min(horizon, 24))
            resid_std = summary.loc[model_name, "RMSE"] if model_name in summary.index else float(series.std())
            future_idx = pd.date_range(series.index[-1], periods=horizon + 1, freq=cfg["freq"])[1:]

            fig = go.Figure()
            hist = series.iloc[-cfg["season_length"] * 3:]
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="History", line=dict(color="#93a1c2")))
            fig.add_trace(go.Scatter(x=future_idx, y=point, mode="lines+markers", name="Forecast", line=dict(color="#6ee7f2", width=3)))
            fig.add_trace(go.Scatter(
                x=list(future_idx) + list(future_idx[::-1]),
                y=list(point + 1.28 * resid_std) + list((point - 1.28 * resid_std)[::-1]),
                fill="toself", fillcolor="rgba(110,231,242,0.15)", line=dict(width=0),
                name="~80% interval", showlegend=True,
            ))
            theme.style_fig(fig, 440)
            st.plotly_chart(fig, width='stretch')

            m1, m2 = st.columns(2)
            m1.metric(f"Forecast at t+{horizon}", f"{point[-1]:,.1f}")
            m2.metric("Backtested RMSE for this model", f"{resid_std:,.1f}")

    theme.footer("05 · Time Series Forecasting")


if __name__ == "__main__":
    main()
