"""Customer Segmentation & Clustering
=======================================
A CRISP-DM-structured unsupervised-learning project with **two** real
datasets, switchable from the sidebar:

* **Mall Customers** — classic demographic segmentation (Age, Income,
  Spending Score).
* **Online Retail RFM** — behavioral segmentation (Recency, Frequency,
  Monetary) derived from ~4,000 real UK online-retail invoices.

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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS = {
    "Mall Customers (demographic)": {
        "path": DATA_DIR / "Mall_Customers.csv",
        "id_col": "CustomerID",
        "features": ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        "source": "Kaggle-style demographic panel (200 shoppers): age, annual income, and a "
                   "1-100 spending score assigned by the mall's loyalty program.",
    },
    "Online Retail RFM (behavioral)": {
        "path": DATA_DIR / "retail_rfm.csv",
        "id_col": "CustomerID",
        "features": ["recency_days", "frequency", "monetary"],
        "log_features": ["frequency", "monetary"],
        "source": "Derived from the UCI 'Online Retail' dataset (a UK online gift retailer, "
                   "Dec 2010-Dec 2011): Recency (days since last order), Frequency (# distinct "
                   "invoices), Monetary (total £ spend) per customer.",
    },
}


def prep_matrix(df: pd.DataFrame, features: list[str], log_features: list[str] | None = None) -> pd.DataFrame:
    """Return the model-ready feature frame: log1p on heavy-tailed monetary/frequency
    columns (standard RFM practice) before the caller standard-scales it."""
    X = df[features].copy()
    for f in (log_features or []):
        if f in X.columns:
            X[f] = np.log1p(X[f].clip(lower=0))
    return X

CLUSTER_COLORS = px.colors.qualitative.Set2


@st.cache_data
def load_data(name: str) -> pd.DataFrame:
    cfg = DATASETS[name]
    return pd.read_csv(cfg["path"])


@st.cache_resource(show_spinner=False)
def fit_kmeans_sweep(X_scaled: np.ndarray, k_max: int = 10):
    inertias, sils = [], []
    ks = list(range(2, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, km.labels_))
    return ks, inertias, sils


def profile_table(df: pd.DataFrame, features: list[str], label_col: str = "cluster") -> pd.DataFrame:
    agg = df.groupby(label_col)[features].mean().round(1)
    agg["n_customers"] = df.groupby(label_col).size()
    agg["share_%"] = (100 * agg["n_customers"] / len(df)).round(1)
    return agg.sort_index()


def persona_label(row: pd.Series, features: list[str]) -> str:
    """Very small heuristic namer so clusters read as personas, not just numbers."""
    z = row[features]
    top = z.idxmax()
    bottom = z.idxmin()
    return f"High {top} / Low {bottom}"


def main():
    theme.apply("🧩", "Customer Segmentation & Clustering")
    theme.hero(
        "🧩", "Customer Segmentation & Clustering",
        "Unsupervised CRISP-DM pipeline comparing K-Means, DBSCAN, and Agglomerative "
        "clustering on two real customer datasets — demographic and behavioral (RFM).",
        ["Unsupervised ML", "K-Means · DBSCAN · Agglomerative", "scikit-learn", "Streamlit"],
    )

    with st.sidebar:
        st.subheader("📁 Dataset")
        dataset_name = st.radio("Choose a segmentation angle", list(DATASETS.keys()))
        cfg = DATASETS[dataset_name]
        df_raw = load_data(dataset_name)
        st.metric("Customers", f"{len(df_raw):,}")
        st.caption(cfg["source"])

    features = cfg["features"]
    df = df_raw.dropna(subset=features).copy()

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Business objective")
        st.markdown(
            f"""
Marketing wants to move from **one-size-fits-all campaigns** to **targeted personas** —
different offers for price-sensitive vs. high-value customers, different win-back
tactics for lapsing customers vs. loyal ones.

**This angle:** *{dataset_name}*. Features used for clustering: {", ".join(f"`{f}`" for f in features)}.

**Success criteria (data science):** maximize silhouette score (cluster cohesion vs.
separation) while keeping segment count small enough to be actionable (2-6 segments).

**Success criteria (business):** every segment must be nameable as a persona a marketer
can design a campaign around.
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Task</div>'
                     '<div class="ds-metric-value">Clustering</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="ds-card"><div class="ds-metric-label">Features</div>'
                     f'<div class="ds-metric-value">{len(features)}</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Primary metric</div>'
                     '<div class="ds-metric-value">Silhouette score</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding --------------------------------------------------------
    with tabs[1]:
        st.subheader("Exploratory data analysis")
        cols = st.columns(len(features))
        for c, f in zip(cols, features):
            fig = px.histogram(df, x=f, nbins=25, title=f)
            theme.style_fig(fig, 300)
            c.plotly_chart(fig, width='stretch')

        if len(features) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(df, x=features[0], y=features[1], opacity=0.6,
                                  title=f"{features[0]} vs {features[1]}")
                theme.style_fig(fig, 380)
                st.plotly_chart(fig, width='stretch')
            with c2:
                fig = px.imshow(df[features].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Feature correlation")
                theme.style_fig(fig, 380)
                st.plotly_chart(fig, width='stretch')

        if "Genre" in df.columns:
            fig = px.histogram(df, x="Genre", color="Genre", title="Gender distribution")
            theme.style_fig(fig, 300)
            st.plotly_chart(fig, width='stretch')

        st.dataframe(df.describe().T, width='stretch')

    # 3. Data Preparation -----------------------------------------------------------
    with tabs[2]:
        st.subheader("Scaling & preparation")
        log_feats = cfg.get("log_features", [])
        if log_feats:
            st.markdown(
                f"**Log-transform** {', '.join(f'`{f}`' for f in log_feats)} first — RFM "
                "monetary/frequency values are heavily right-skewed (a handful of "
                "wholesale-scale buyers dwarf everyone else); `log1p` pulls them back "
                "toward a symmetric distribution *before* scaling, which is standard "
                "RFM segmentation practice and prevents those outliers from single-handedly "
                "defining cluster centroids."
            )
        st.markdown(
            """
K-Means and DBSCAN are distance-based, so every feature must be on a comparable
scale — a `StandardScaler` (zero mean, unit variance) is fit here and reused for
every model in this tab, preventing any one feature (e.g. `monetary`, which can span
thousands of pounds) from dominating the Euclidean distance purely due to units.
            """
        )
        X_prepped = prep_matrix(df, features, log_feats)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_prepped)
        scaled_df = pd.DataFrame(X_scaled, columns=features)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Before scaling (post log-transform if applicable)")
            st.dataframe(X_prepped.describe().T[["mean", "std", "min", "max"]].round(2), width='stretch')
        with c2:
            st.caption("After scaling")
            st.dataframe(scaled_df.describe().T[["mean", "std", "min", "max"]].round(2), width='stretch')

    # 4. Modeling ---------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Choosing k — elbow method & silhouette analysis")
        log_feats = cfg.get("log_features", [])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(prep_matrix(df, features, log_feats))
        ks, inertias, sils = fit_kmeans_sweep(X_scaled, k_max=min(10, len(df) - 1))

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(x=ks, y=inertias, markers=True, labels={"x": "k", "y": "inertia"},
                          title="Elbow method (inertia vs k)")
            theme.style_fig(fig, 340)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.line(x=ks, y=sils, markers=True, labels={"x": "k", "y": "silhouette"},
                          title="Silhouette score vs k")
            theme.style_fig(fig, 340)
            st.plotly_chart(fig, width='stretch')

        best_k = ks[int(np.argmax(sils))]
        st.info(f"Silhouette-optimal k = **{best_k}** (highest separation/cohesion trade-off).", icon="🧠")

        st.markdown("#### Pick k and an algorithm")
        c1, c2 = st.columns(2)
        with c1:
            k = st.slider("Number of clusters (k)", 2, min(10, len(df) - 1), best_k)
        with c2:
            algo = st.selectbox("Algorithm", ["K-Means", "Agglomerative (Ward)", "DBSCAN"])

        if algo == "K-Means":
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(X_scaled)
        elif algo == "Agglomerative (Ward)":
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X_scaled)
        else:
            eps = st.slider("DBSCAN eps", 0.2, 2.0, 0.6, 0.05)
            min_samples = st.slider("DBSCAN min_samples", 3, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

        df_clustered = df.copy()
        df_clustered["cluster"] = labels
        n_found = len(set(labels) - {-1})
        noise = int((labels == -1).sum())
        sil = silhouette_score(X_scaled, labels) if n_found > 1 else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("Clusters found", n_found)
        m2.metric("Noise points (DBSCAN only)", noise)
        m3.metric("Silhouette score", f"{sil:.3f}" if sil == sil else "n/a")

        if len(features) >= 2:
            fig = px.scatter(
                df_clustered, x=features[0], y=features[1], color=df_clustered["cluster"].astype(str),
                title=f"{algo} — {features[0]} vs {features[1]}", color_discrete_sequence=CLUSTER_COLORS,
            )
            theme.style_fig(fig, 420)
            st.plotly_chart(fig, width='stretch')

        st.session_state["_cluster_result"] = (df_clustered, features, algo, scaler, model if algo == "K-Means" else None, log_feats)

    # 5. Evaluation ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Cluster profiles")
        if "_cluster_result" not in st.session_state:
            st.warning("Visit the Modeling tab first to fit a clustering model.")
        else:
            df_clustered, feats, algo, scaler, km_model, _log_feats = st.session_state["_cluster_result"]
            profile = profile_table(df_clustered, feats)
            st.dataframe(profile, width='stretch')

            fig = go.Figure()
            norm = (profile[feats] - profile[feats].min()) / (profile[feats].max() - profile[feats].min() + 1e-9)
            for cluster_id, row in norm.iterrows():
                fig.add_trace(go.Scatterpolar(r=row.values, theta=feats, fill="toself", name=f"Cluster {cluster_id}"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Segment shape (normalized)")
            theme.style_fig(fig, 440)
            st.plotly_chart(fig, width='stretch')

            fig = px.bar(profile.reset_index(), x="cluster", y="n_customers", color="cluster",
                         color_discrete_sequence=CLUSTER_COLORS, title="Segment sizes")
            theme.style_fig(fig, 320)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

    # 6. Deployment --------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Assign a new customer to a segment")
        if "_cluster_result" not in st.session_state:
            st.warning("Visit the Modeling tab first to fit a clustering model (K-Means recommended for live scoring).")
        else:
            df_clustered, feats, algo, scaler, km_model, log_feats = st.session_state["_cluster_result"]
            if km_model is None:
                st.info("Live scoring needs a centroid-based model — switch to **K-Means** in the Modeling tab.", icon="ℹ️")
            else:
                inputs = {}
                cols = st.columns(len(feats))
                for c, f in zip(cols, feats):
                    lo, hi = float(df[f].min()), float(df[f].max())
                    inputs[f] = c.slider(f, lo, hi, float(df[f].median()))

                if st.button("🧩 Assign segment", type="primary"):
                    x = scaler.transform(prep_matrix(pd.DataFrame([inputs]), feats, log_feats))
                    cluster_id = int(km_model.predict(x)[0])
                    profile = profile_table(df_clustered, feats)
                    persona = persona_label(profile.loc[cluster_id], feats)
                    st.success(f"Assigned to **Cluster {cluster_id}** — *{persona}*", icon="🎯")
                    st.dataframe(profile.loc[[cluster_id]], width='stretch')

    theme.footer("02 · Customer Segmentation & Clustering")


if __name__ == "__main__":
    main()
