"""Fraud & Anomaly Detection
==============================
CRISP-DM project on the real Kaggle "Credit Card Fraud Detection" dataset
(ULB Machine Learning Group) — a severely imbalanced classification task
reframed as an anomaly-detection problem: 492 confirmed frauds among
~14,500 transactions (~3.4% fraud rate here, vs. ~0.17% in the full
284,807-row original — subsampled for a fast, repo-friendly demo while
preserving every fraud case).

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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "data" / "card_transactions.csv"
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["Amount", "scaled_time"]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["scaled_time"] = (df["Time"] - df["Time"].mean()) / df["Time"].std()
    return df


@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    ).fit(X_train_s, y_train)

    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42).fit(X_train_s, y_train)

    contamination = min(0.5, max(0.001, y_train.mean()))
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
    ).fit(X_train_s[y_train.values == 0])  # trained on "normal" only, like a true anomaly detector

    scores = {
        "Random Forest (supervised)": rf.predict_proba(X_test_s)[:, 1],
        "Logistic Regression (supervised)": lr.predict_proba(X_test_s)[:, 1],
        "Isolation Forest (unsupervised)": -iso.score_samples(X_test_s),  # higher = more anomalous
    }
    # normalize isolation forest score to [0, 1] for comparable thresholds/plots
    iso_raw = scores["Isolation Forest (unsupervised)"]
    scores["Isolation Forest (unsupervised)"] = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

    models = {"Random Forest (supervised)": rf, "Logistic Regression (supervised)": lr, "Isolation Forest (unsupervised)": iso}
    return models, scaler, scores, y_test.reset_index(drop=True), X_test.reset_index(drop=True)


def metrics_table(scores: dict, y_test: pd.Series) -> pd.DataFrame:
    rows = {}
    for name, s in scores.items():
        pr_auc = average_precision_score(y_test, s)
        roc_auc = roc_auc_score(y_test, s)
        prec, rec, thr = precision_recall_curve(y_test, s)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
        best_thr = thr[best_idx] if len(thr) else 0.5
        preds = (s >= best_thr).astype(int)
        rows[name] = {
            "PR-AUC": pr_auc, "ROC-AUC": roc_auc,
            "Best F1": f1s[best_idx] if len(thr) else float("nan"),
            "Threshold @ best F1": best_thr,
            "Precision @ best F1": prec[best_idx] if len(thr) else float("nan"),
            "Recall @ best F1": rec[best_idx] if len(thr) else float("nan"),
        }
    return pd.DataFrame(rows).T


def main():
    theme.apply("🛡️", "Fraud & Anomaly Detection")
    theme.hero(
        "🛡️", "Fraud & Anomaly Detection",
        "CRISP-DM anomaly-detection pipeline on real, PCA-anonymized Kaggle credit-card "
        "transactions — supervised (Random Forest, Logistic Regression) vs. unsupervised "
        "(Isolation Forest), evaluated with PR-AUC on a ~3.4% fraud rate.",
        ["Imbalanced classification", "Isolation Forest", "PR-AUC over Accuracy", "Streamlit"],
    )

    df = load_data()

    with st.sidebar:
        st.subheader("📁 Dataset")
        st.metric("Transactions", f"{len(df):,}")
        st.metric("Confirmed frauds", f"{int(df['Class'].sum()):,}")
        st.metric("Fraud rate", f"{df['Class'].mean()*100:.2f}%")
        st.caption(
            "Source: Kaggle 'Credit Card Fraud Detection' (ULB Machine Learning Group). "
            "Features V1-V28 are PCA components of the original (undisclosed, "
            "privacy-protected) transaction attributes; only `Time` and `Amount` are raw."
        )

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Business objective")
        st.markdown(
            """
Every missed fraud (**false negative**) costs the full transaction value plus
chargeback fees; every false fraud alarm (**false positive**) costs customer trust
and support overhead — but the two costs are *not symmetric*, and accuracy is a
**trap metric** here: a model that predicts "not fraud" for every transaction would
score ~96.6% accuracy on this dataset while catching zero fraud.

**Success criteria (data science):** maximize **PR-AUC** (precision-recall area under
curve) — the right lens for severe class imbalance, unlike ROC-AUC or accuracy.

**Success criteria (business):** a tunable decision threshold that lets risk teams
trade precision for recall along a real cost curve (see Evaluation tab).
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Task</div>'
                     '<div class="ds-metric-value">Imbalanced classification</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="ds-card"><div class="ds-metric-label">Positive rate</div>'
                     f'<div class="ds-metric-value">{df["Class"].mean()*100:.2f}%</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Primary metric</div>'
                     '<div class="ds-metric-value">PR-AUC</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding ---------------------------------------------------------
    with tabs[1]:
        st.subheader("Exploratory data analysis")
        c1, c2 = st.columns(2)
        with c1:
            counts = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
            fig = px.bar(counts, title="Class balance (log scale)", log_y=True, color=counts.index,
                         color_discrete_map={"Legitimate": "#6ee7f2", "Fraud": "#fb7185"})
            theme.style_fig(fig, 340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.box(df, x="Class", y="Amount", color="Class", log_y=True,
                         title="Transaction amount by class (log scale)",
                         color_discrete_map={0: "#6ee7f2", 1: "#fb7185"})
            theme.style_fig(fig, 340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        st.markdown("#### 2-component PCA projection of the 28 anonymized features")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(StandardScaler().fit_transform(df[V_COLS]))
        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        plot_df["Class"] = df["Class"].map({0: "Legitimate", 1: "Fraud"}).values
        sample = pd.concat([plot_df[plot_df["Class"] == "Fraud"],
                             plot_df[plot_df["Class"] == "Legitimate"].sample(3000, random_state=1)])
        fig = px.scatter(sample, x="PC1", y="PC2", color="Class", opacity=0.55,
                          color_discrete_map={"Legitimate": "#6ee7f2", "Fraud": "#fb7185"},
                          title=f"PCA(V1..V28) — {pca.explained_variance_ratio_.sum()*100:.0f}% variance explained")
        theme.style_fig(fig, 440)
        st.plotly_chart(fig, width='stretch')
        st.caption("Fraud points (red) already separate visibly from legitimate traffic even in "
                   "just 2 dimensions — a strong signal this problem is learnable.")

    # 3. Data Preparation ------------------------------------------------------------
    with tabs[2]:
        st.subheader("Preparation & split strategy")
        st.markdown(
            """
1. `Time` (seconds since first transaction) is standardized to `scaled_time`; `V1..V28`
   are already PCA components from the original data collection, `Amount` is raw.
2. **Stratified 70/30 train/test split** — preserves the ~3.4% fraud rate in both
   splits (a plain random split can accidentally starve the test set of positives).
3. `StandardScaler` is fit **only on the training split**, then applied to test —
   zero leakage of test-set statistics into training.
4. **No SMOTE/oversampling of the test set** — synthetic minority samples must never
   appear in evaluation data, or metrics become meaninglessly optimistic. Class
   imbalance is instead handled *at the model* via `class_weight="balanced"` (RF, LR)
   or by training Isolation Forest exclusively on the majority ("normal") class, the
   textbook-correct way to build an anomaly detector.
            """
        )
        st.dataframe(df[["Time", "scaled_time", "Amount", "Class"] + V_COLS[:4]].head(10), width='stretch')

    # 4. Modeling -----------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Model tournament: supervised vs. unsupervised")
        models, scaler, scores, y_test, X_test = train_models(df)
        st.caption(
            "Random Forest & Logistic Regression see fraud labels during training (supervised); "
            "Isolation Forest never does — it only learns what 'normal' looks like and flags "
            "deviations (unsupervised anomaly detection), the more realistic setup when fraud "
            "labels are scarce or delayed in production."
        )
        mt = metrics_table(scores, y_test)
        st.dataframe(
            mt.style.format("{:.3f}").background_gradient(subset=["PR-AUC"], cmap="Greens"),
            width='stretch',
        )

        st.markdown("#### Precision-Recall curves")
        fig = go.Figure()
        for name, s in scores.items():
            prec, rec, _ = precision_recall_curve(y_test, s)
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=name))
        baseline = y_test.mean()
        fig.add_hline(y=baseline, line_dash="dash", line_color="#93a1c2",
                      annotation_text=f"random baseline ({baseline:.3f})")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        theme.style_fig(fig, 440)
        st.plotly_chart(fig, width='stretch')

        rf = models["Random Forest (supervised)"]
        importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(12)
        fig = px.bar(importances[::-1], orientation="h", title="Top 12 features — Random Forest importance")
        theme.style_fig(fig, 380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # 5. Evaluation -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Cost-sensitive threshold tuning")
        models, scaler, scores, y_test, X_test = train_models(df)
        model_choice = st.selectbox("Model", list(scores.keys()))
        s = scores[model_choice]

        c1, c2 = st.columns(2)
        fn_cost = c1.slider("Cost of a missed fraud (false negative, $)", 10, 2000, 500, 10)
        fp_cost = c2.slider("Cost of a false alarm (false positive, $)", 1, 200, 15, 1)

        thresholds = np.linspace(0.01, 0.99, 99)
        costs = []
        for t in thresholds:
            preds = (s >= t).astype(int)
            fn = int(((preds == 0) & (y_test == 1)).sum())
            fp = int(((preds == 1) & (y_test == 0)).sum())
            costs.append(fn * fn_cost + fp * fp_cost)
        best_t = thresholds[int(np.argmin(costs))]

        fig = px.line(x=thresholds, y=costs, labels={"x": "Decision threshold", "y": "Total expected cost ($)"},
                      title=f"Expected cost vs. threshold — optimal at {best_t:.2f}")
        fig.add_vline(x=best_t, line_dash="dash", line_color="#34d399")
        theme.style_fig(fig, 380)
        st.plotly_chart(fig, width='stretch')

        preds = (s >= best_t).astype(int)
        cm = confusion_matrix(y_test, preds)
        fig = px.imshow(cm, text_auto=True, x=["Pred: Legit", "Pred: Fraud"], y=["True: Legit", "True: Fraud"],
                         color_continuous_scale="Tealgrn", title=f"Confusion matrix @ cost-optimal threshold ({best_t:.2f})")
        theme.style_fig(fig, 380)
        st.plotly_chart(fig, width='stretch')

        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud caught (recall)", f"{cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")
        m2.metric("Precision", f"{cm[1,1]/max(1,(cm[0,1]+cm[1,1]))*100:.1f}%")
        m3.metric("Minimum expected cost", f"${min(costs):,.0f}")

    # 6. Deployment -----------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Score a transaction")
        models, scaler, scores, y_test, X_test = train_models(df)
        rf = models["Random Forest (supervised)"]

        mode = st.radio("Pick a transaction", ["Sample from test set", "Manual (Amount + top signals)"], horizontal=True)

        if mode == "Sample from test set":
            idx = st.number_input("Test-set row index", 0, len(X_test) - 1, 0, 1)
            row = X_test.iloc[idx]
            true_label = y_test.iloc[idx]
            st.caption(f"Ground truth for this row: {'🚨 Fraud' if true_label == 1 else '✅ Legitimate'} (for teaching purposes only)")
            x = scaler.transform(pd.DataFrame([row])[FEATURES])
        else:
            top_feats = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(4).index.tolist()
            defaults = X_test[FEATURES].median()
            inputs = defaults.to_dict()
            cols = st.columns(len(top_feats) + 1)
            inputs["Amount"] = cols[0].slider("Amount ($)", 0.0, float(df["Amount"].quantile(0.99)), float(df["Amount"].median()))
            for c, f in zip(cols[1:], top_feats):
                lo, hi = float(df[f].quantile(0.01)), float(df[f].quantile(0.99))
                inputs[f] = c.slider(f, lo, hi, float(df[f].median()))
            x = scaler.transform(pd.DataFrame([inputs])[FEATURES])

        if st.button("🛡️ Score transaction", type="primary"):
            proba = float(rf.predict_proba(x)[0, 1])
            iso_score = float(models["Isolation Forest (unsupervised)"].score_samples(x)[0])
            m1, m2 = st.columns(2)
            m1.metric("Random Forest fraud probability", f"{proba*100:.1f}%")
            m2.metric("Isolation Forest anomaly score", f"{-iso_score:.3f}", help="Higher = more anomalous")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                title={"text": "Fraud risk"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#fb7185" if proba > 0.5 else "#34d399"},
                       "steps": [{"range": [0, 30], "color": "#0f1729"},
                                 {"range": [30, 70], "color": "#1a2440"},
                                 {"range": [70, 100], "color": "#2a1a24"}]},
            ))
            theme.style_fig(fig, 300)
            st.plotly_chart(fig, width='stretch')

    theme.footer("04 · Fraud & Anomaly Detection")


if __name__ == "__main__":
    main()
