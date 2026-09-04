"""CRISP-DM Master's Curriculum
==================================
A slower, textbook-paced walk through **every** CRISP-DM phase on a single,
historically famous dataset — Fisher's Iris flowers (1936) — with a concept
check at each phase and a running curriculum score in the sidebar.

Where project 07 teaches four *isolated* concepts and projects 01-06 each
go deep on *one* technique, this project's teaching point is different:
it puts **unsupervised clustering and supervised classification side by
side on the exact same data**, so the difference between "discover
structure" and "predict a known label" is directly visible rather than
spread across separate projects.

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
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


@st.cache_data
def load_data():
    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(bunch.target_names)))
    return df, list(bunch.target_names)


def concept_check(phase_key: str, question: str, options: list[str], correct_idx: int, explain: str):
    """A small, reusable per-phase quiz widget that tallies into session_state."""
    st.markdown(f"##### 🧠 Concept check")
    choice = st.radio(question, options, index=None, key=f"cc_{phase_key}")
    if choice is not None:
        is_right = options.index(choice) == correct_idx
        st.session_state.setdefault("curriculum_score", {})[phase_key] = is_right
        if is_right:
            st.success(f"Correct — {explain}", icon="✅")
        else:
            st.error(f"Not quite — {explain}", icon="❌")


def main():
    theme.apply("📚", "CRISP-DM Master's Curriculum")
    theme.hero(
        "📚", "CRISP-DM Master's Curriculum",
        "A textbook-paced walk through every CRISP-DM phase on Fisher's famous 1936 Iris "
        "dataset — with clustering and classification placed side by side on the same data, "
        "and a concept check at every phase.",
        ["Textbook curriculum", "Clustering vs. Classification", "Concept checks", "Streamlit"],
    )

    df, species_names = load_data()
    scores = st.session_state.get("curriculum_score", {})

    with st.sidebar:
        st.subheader("📁 Dataset")
        st.metric("Flowers measured", len(df))
        st.metric("Species", len(species_names))
        st.caption(
            "R.A. Fisher's Iris dataset (1936) — 150 real iris flowers, 4 physical "
            "measurements each, 3 species. Arguably the single most-used dataset in the "
            "history of statistics and machine learning teaching."
        )
        st.divider()
        st.subheader("🎓 Curriculum progress")
        answered = len(scores)
        correct = sum(scores.values())
        st.progress(answered / 6 if answered else 0.0)
        st.caption(f"{answered}/6 concept checks answered · {correct} correct")

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Chapter 1 — Business Understanding")
        st.markdown(
            """
In 1936, statistician and biologist **Ronald Fisher** published a method for
distinguishing three species of iris flower using just four physical
measurements — sepal length/width and petal length/width — instead of
requiring a trained botanist to identify the species by eye. That's the
essence of *every* applied data science problem since: **replace an
expensive, slow, or scarce expert judgment with a fast, cheap, repeatable
measurement-based rule.**

**The business question:** given only the four measurements, which species
is this flower? **Why it's a good business problem:** the labels
(species) are well-defined, the features are cheap and fast to collect
(a ruler), and getting it right or wrong has a clear, checkable answer.
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Task</div>'
                     '<div class="ds-metric-value">Classification</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="ds-card"><div class="ds-metric-label">Classes</div>'
                     f'<div class="ds-metric-value">{", ".join(species_names)}</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Features</div>'
                     '<div class="ds-metric-value">4 physical measurements</div></div>', unsafe_allow_html=True)

        concept_check(
            "business",
            "Why is 'identify the iris species' a *good* business problem to formalize with data?",
            ["Because flowers are pretty", "Because the labels are well-defined and the features are cheap to measure",
             "Because it requires no data at all", "Because it has no possible wrong answer"],
            1,
            "well-defined labels + cheap, repeatable measurement are exactly what make a problem tractable for data science.",
        )

    # 2. Data Understanding ---------------------------------------------------------
    with tabs[1]:
        st.subheader("Chapter 2 — Data Understanding")
        st.markdown("Before modeling anything, *look at the data*. Do the species actually separate visually?")
        fig = px.scatter_matrix(df, dimensions=FEATURES, color="species", title="Pairwise feature relationships by species")
        theme.style_fig(fig, 620)
        st.plotly_chart(fig, width='stretch')

        st.dataframe(df.groupby("species")[FEATURES].mean().round(2), width='stretch')

        concept_check(
            "data_understanding",
            "Looking at the pairwise scatter plots above, which pair of features separates the species most cleanly?",
            ["sepal length vs sepal width", "petal length vs petal width", "None of them separate at all", "You can't tell from a chart"],
            1,
            "petal length/width form the cleanest, almost non-overlapping clusters — sepal measurements overlap much more between versicolor and virginica.",
        )

    # 3. Data Preparation ------------------------------------------------------------
    with tabs[2]:
        st.subheader("Chapter 3 — Data Preparation")
        st.markdown(
            f"""
**Missing values:** {df[FEATURES].isna().sum().sum()} (this dataset, like most textbook
datasets, is already clean — real-world data rarely is, which is *why* [project 04](../04_fraud_anomaly_detection)
and others in this portfolio spend a whole tab on cleaning).

**Scaling:** the four features are all in centimeters and similar magnitude here,
but K-Means (used in the next chapter) is still distance-based, so we scale
anyway — good habit, zero cost when it's already unnecessary, essential when it isn't.

**Split:** a **stratified** 70/30 train/test split, preserving each
species' share in both halves.
            """
        )
        X_train, X_test, y_train, y_test = train_test_split(
            df[FEATURES], df["target"], test_size=0.3, random_state=42, stratify=df["target"]
        )
        st.code(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.3, random_state=42, stratify=y\n"
            ")",
            language="python",
        )
        st.session_state["_iris_split"] = (X_train, X_test, y_train, y_test)

        concept_check(
            "data_prep",
            "Why stratify the split by species instead of a plain random split?",
            ["It trains faster", "It guarantees every species is proportionally represented in both train and test",
             "It removes the need for scaling", "It's required by every scikit-learn function"],
            1,
            "a plain random split *can* accidentally under- or over-represent a class in the test set, especially with few examples per class.",
        )

    # 4. Modeling: clustering vs classification, side by side -----------------------------
    with tabs[3]:
        st.subheader("Chapter 4 — Modeling: two fundamentally different questions")
        X_train, X_test, y_train, y_test = st.session_state.get("_iris_split", train_test_split(
            df[FEATURES], df["target"], test_size=0.3, random_state=42, stratify=df["target"]))

        st.markdown(
            """
**Unsupervised (clustering):** "if I *didn't* know the species labels, would
the measurements alone reveal 3 natural groups?" — K-Means never sees `y`.

**Supervised (classification):** "given the labels, learn a rule to predict
species on *new* flowers." — the model directly optimizes against `y`.
            """
        )

        scaler = StandardScaler().fit(df[FEATURES])
        X_scaled_full = scaler.transform(df[FEATURES])
        km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled_full)
        ari = adjusted_rand_score(df["target"], km.labels_)

        c1, c2 = st.columns(2)
        with c1:
            plot_df = df.copy()
            plot_df["cluster"] = km.labels_.astype(str)
            fig = px.scatter(plot_df, x="petal length (cm)", y="petal width (cm)", color="cluster",
                              title=f"K-Means clusters (blind to species) — ARI={ari:.2f} vs. true species")
            theme.style_fig(fig, 380)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.scatter(df, x="petal length (cm)", y="petal width (cm)", color="species",
                              title="True species (for comparison)")
            theme.style_fig(fig, 380)
            st.plotly_chart(fig, width='stretch')
        st.caption("Adjusted Rand Index (ARI) measures cluster-vs-label agreement, corrected for chance "
                   "(1.0 = perfect agreement, 0.0 = random). K-Means finds the real structure almost perfectly here.")

        st.markdown("#### Now the supervised side — 3 classifiers compared")
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        }
        results, fitted = {}, {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            results[name] = accuracy_score(y_test, pred)
            fitted[name] = clf
        st.session_state["_iris_models"] = fitted
        res_df = pd.Series(results, name="Test accuracy").sort_values(ascending=False).to_frame()
        st.dataframe(res_df.style.format("{:.3f}").background_gradient(cmap="Greens"), width='stretch')

        concept_check(
            "modeling",
            "K-Means (unsupervised) recovers species groupings almost perfectly here without ever seeing labels. What does that tell you?",
            ["Clustering is always better than classification", "The species are so well-separated in feature space that structure alone reveals them",
             "K-Means secretly used the labels", "This never happens in real data"],
            1,
            "when classes are naturally well-separated in feature space, unsupervised structure and supervised labels agree closely — that's the ideal case, not the norm (real classes usually overlap more).",
        )

    # 5. Evaluation -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Chapter 5 — Evaluation")
        fitted = st.session_state.get("_iris_models")
        X_train, X_test, y_train, y_test = st.session_state.get("_iris_split", (None, None, None, None))
        if not fitted or X_test is None:
            st.warning("Visit the Modeling tab first.")
        else:
            best_name = max(fitted, key=lambda n: accuracy_score(y_test, fitted[n].predict(X_test)))
            best = fitted[best_name]
            pred = best.predict(X_test)
            acc = accuracy_score(y_test, pred)
            st.success(f"Best model: **{best_name}** ({acc:.1%} test accuracy)", icon="🏆")

            cm = confusion_matrix(y_test, pred)
            fig = px.imshow(cm, text_auto=True, x=species_names, y=species_names,
                             color_continuous_scale="Tealgrn", title=f"Confusion matrix — {best_name}",
                             labels=dict(x="Predicted", y="True"))
            theme.style_fig(fig, 400)
            st.plotly_chart(fig, width='stretch')

            if best_name == "Decision Tree":
                imp = pd.Series(best.feature_importances_, index=FEATURES).sort_values()
                fig = px.bar(imp, orientation="h", title="Feature importance")
                theme.style_fig(fig, 300); fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')

            concept_check(
                "evaluation",
                "With 3 balanced classes (50 flowers each), is accuracy a trustworthy headline metric here?",
                ["No, never trust accuracy", "Yes — with balanced classes accuracy isn't misleading, unlike the imbalanced case in project 04",
                 "Only if the model is a neural network", "Accuracy only works for regression"],
                1,
                "accuracy misleads specifically under class imbalance (see project 04's fraud case); with roughly equal classes here it's a fair, simple headline number.",
            )

    # 6. Deployment: live predictor + synthesis --------------------------------------------
    with tabs[5]:
        st.subheader("Chapter 6 — Deployment & Synthesis")
        fitted = st.session_state.get("_iris_models")
        if not fitted:
            st.warning("Visit the Modeling tab first.")
        else:
            st.markdown("#### Identify a flower")
            cols = st.columns(4)
            inputs = {}
            for c, f in zip(cols, FEATURES):
                lo, hi = float(df[f].min()), float(df[f].max())
                inputs[f] = c.slider(f, lo, hi, float(df[f].median()))
            model_name = st.selectbox("Model", list(fitted.keys()))

            if st.button("🌸 Identify species", type="primary"):
                x = pd.DataFrame([inputs])[FEATURES]
                model = fitted[model_name]
                pred = int(model.predict(x)[0])
                st.success(f"Predicted species: **{species_names[pred]}**", icon="🌸")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x)[0]
                    fig = px.bar(x=species_names, y=proba, title="Class probabilities")
                    theme.style_fig(fig, 300); fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')

            st.markdown("---")
            st.markdown("#### 📜 Synthesis")
            correct = sum(st.session_state.get("curriculum_score", {}).values())
            answered = len(st.session_state.get("curriculum_score", {}))
            st.markdown(
                f"""
Across six chapters: a well-posed business question (cheap measurements,
clear labels) → visual confirmation the classes are separable → a
leakage-safe stratified split → clustering *and* classification compared
head-to-head on identical data → an honest accuracy readout backed by a
confusion matrix → a live, usable predictor.

**Your concept-check score: {correct}/{answered} answered correctly.**

This is the same six-phase discipline every other project in this
portfolio follows — [01](../01_nyc_taxi_trip_duration) through
[09](../09_data_science_skills_lab) — applied here at textbook pace, on
the dataset that arguably started the whole field of statistical
classification.
                """
            )

    theme.footer("10 · CRISP-DM Master's Curriculum")


if __name__ == "__main__":
    main()
