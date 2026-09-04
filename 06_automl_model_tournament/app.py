"""AutoML Model Tournament
=============================
A CRISP-DM project that automates what a data scientist would otherwise do
by hand: try many candidate algorithms, tune each one's hyperparameters,
and rank them fairly — the core idea behind AutoML tools (AutoGluon,
H2O AutoML, auto-sklearn), reimplemented here with plain scikit-learn so
the whole thing installs in seconds and runs on any laptop.

Works across **three real, classic datasets** spanning both classification
and regression, from scikit-learn's own bundled real-world data (Wisconsin
Diagnostic Breast Cancer, Italian wine cultivar chemistry, and Pima-adjacent
diabetes progression) — no download required, fully offline-reproducible.

Run:
    streamlit run app.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    r2_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

TASKS = {
    "Breast Cancer Diagnosis (classification)": {
        "loader": load_breast_cancer, "type": "classification",
        "source": "Wisconsin Diagnostic Breast Cancer dataset (real, de-identified digitized "
                   "cell-nuclei measurements from 569 patient biopsies; UCI ML Repository, "
                   "bundled in scikit-learn). Target: malignant vs. benign.",
    },
    "Wine Cultivar Identification (classification)": {
        "loader": load_wine, "type": "classification",
        "source": "Real chemical analysis of 178 wines grown in the same Italian region from "
                   "3 different cultivars (UCI ML Repository, bundled in scikit-learn). "
                   "Target: which of the 3 cultivars produced the wine.",
    },
    "Diabetes Progression (regression)": {
        "loader": load_diabetes, "type": "regression",
        "source": "Real diabetes patient data (442 patients): 10 baseline physiological "
                   "variables and a quantitative measure of disease progression one year "
                   "later (UCI ML Repository, bundled in scikit-learn).",
    },
}

CLASSIFIERS = {
    "Logistic Regression": (LogisticRegression(max_iter=3000), {"model__C": uniform(0.01, 10)}),
    "Random Forest": (RandomForestClassifier(random_state=42), {
        "model__n_estimators": randint(100, 300), "model__max_depth": randint(3, 15)}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        "model__n_estimators": randint(50, 200), "model__max_depth": randint(2, 5),
        "model__learning_rate": uniform(0.02, 0.2)}),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {"model__n_neighbors": randint(3, 20)}),
    "Support Vector Machine": (SVC(probability=True, random_state=42), {
        "model__C": uniform(0.1, 10), "model__gamma": uniform(0.001, 0.1)}),
    "Naive Bayes": (GaussianNB(), {}),
}

REGRESSORS = {
    "Ridge Regression": (Ridge(), {"model__alpha": uniform(0.01, 10)}),
    "Random Forest": (RandomForestRegressor(random_state=42), {
        "model__n_estimators": randint(100, 300), "model__max_depth": randint(3, 15)}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=42), {
        "model__n_estimators": randint(50, 200), "model__max_depth": randint(2, 5),
        "model__learning_rate": uniform(0.02, 0.2)}),
    "K-Nearest Neighbors": (KNeighborsRegressor(), {"model__n_neighbors": randint(3, 20)}),
    "Support Vector Machine": (SVR(), {"model__C": uniform(0.1, 10), "model__gamma": uniform(0.001, 0.1)}),
}


@st.cache_data
def load_task_data(task_name: str):
    cfg = TASKS[task_name]
    bunch = cfg["loader"](as_frame=True)
    X, y = bunch.data, bunch.target
    df = bunch.frame.copy()
    target_names = getattr(bunch, "target_names", None)
    return X, y, df, target_names, list(X.columns)


@st.cache_resource(show_spinner=True)
def run_tournament(task_name: str, n_iter: int = 10):
    cfg = TASKS[task_name]
    X, y, df, target_names, feature_names = load_task_data(task_name)
    is_clf = cfg["type"] == "classification"
    candidates = CLASSIFIERS if is_clf else REGRESSORS

    strat = y if is_clf else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

    leaderboard, fitted = [], {}
    for name, (estimator, param_dist) in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        t0 = time.perf_counter()
        if param_dist:
            search = RandomizedSearchCV(
                pipe, param_dist, n_iter=n_iter, cv=5, random_state=42, n_jobs=-1,
                scoring="roc_auc_ovr" if (is_clf and len(np.unique(y)) > 2) else ("roc_auc" if is_clf else "r2"),
            )
            search.fit(X_train, y_train)
            best_pipe, best_params, cv_score = search.best_estimator_, search.best_params_, search.best_score_
        else:
            pipe.fit(X_train, y_train)
            best_pipe, best_params = pipe, {}
            from sklearn.model_selection import cross_val_score
            cv_score = cross_val_score(pipe, X_train, y_train, cv=5,
                                        scoring="roc_auc_ovr" if (is_clf and len(np.unique(y)) > 2) else ("roc_auc" if is_clf else "r2")).mean()
        elapsed = time.perf_counter() - t0

        if is_clf:
            pred = best_pipe.predict(X_test)
            test_score = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")
            leaderboard.append({"Model": name, "CV score (ROC-AUC)": cv_score, "Test accuracy": test_score,
                                 "Test F1 (weighted)": f1, "Tuning time (s)": elapsed, "Best params": str(best_params)})
        else:
            pred = best_pipe.predict(X_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            leaderboard.append({"Model": name, "CV score (R2)": cv_score, "Test R2": r2,
                                 "Test MAE": mae, "Tuning time (s)": elapsed, "Best params": str(best_params)})
        fitted[name] = best_pipe

    lb = pd.DataFrame(leaderboard).sort_values("CV score (ROC-AUC)" if is_clf else "CV score (R2)", ascending=False).reset_index(drop=True)
    return lb, fitted, (X_train, X_test, y_train, y_test), is_clf, target_names, feature_names


def main():
    theme.apply("🤖", "AutoML Model Tournament")
    theme.hero(
        "🤖", "AutoML Model Tournament",
        "CRISP-DM AutoML pipeline: automatically tunes and ranks 5-6 candidate algorithms "
        "per task via cross-validated random search — the core AutoML idea, built with "
        "plain scikit-learn on three real, classic datasets.",
        ["AutoML", "RandomizedSearchCV", "Classification + Regression", "Streamlit"],
    )

    with st.sidebar:
        st.subheader("📁 Task")
        task_name = st.radio("Choose a task", list(TASKS.keys()))
        cfg = TASKS[task_name]
        X, y, df, target_names, feature_names = load_task_data(task_name)
        st.metric("Rows", f"{len(df):,}")
        st.metric("Features", f"{len(feature_names):,}")
        st.caption(cfg["source"])
        n_iter = st.slider("Hyperparameter search budget (n_iter)", 5, 25, 10,
                            help="Random search trials per model — higher = better tuning, slower.")

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Business objective")
        st.markdown(
            """
Hand-tuning one model at a time — pick an algorithm, guess hyperparameters,
evaluate, repeat — doesn't scale across the dozens of modeling problems a
data team actually faces. **AutoML** automates that search: try a broad
slate of algorithm families, tune each with cross-validated random search,
and surface a fair leaderboard so a human only has to review the winner
(and its runner-ups) rather than run the whole search by hand.

> This app is a **from-scratch AutoML implementation on plain
> scikit-learn** (`Pipeline` + `RandomizedSearchCV`), not a wrapper around
> the AutoGluon package — a deliberate, transparent scope choice that keeps
> the whole portfolio installable in seconds with no extra heavyweight
> dependency, while demonstrating exactly the same core idea AutoGluon
> automates: multi-algorithm, multi-hyperparameter search with a
> leaderboard.

**Success criteria:** every model is compared on **identical
cross-validated folds** (no model gets an easier evaluation than another),
scored with a metric appropriate to the task (ROC-AUC for classification,
R² for regression, never raw accuracy alone).
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="ds-card"><div class="ds-metric-label">Task type</div>'
                     f'<div class="ds-metric-value">{cfg["type"].title()}</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="ds-card"><div class="ds-metric-label">Candidates</div>'
                     f'<div class="ds-metric-value">{len(CLASSIFIERS if cfg["type"]=="classification" else REGRESSORS)} algorithms</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Search method</div>'
                     '<div class="ds-metric-value">5-fold CV random search</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding ---------------------------------------------------------
    with tabs[1]:
        st.subheader("Exploratory data analysis")
        if cfg["type"] == "classification":
            counts = y.value_counts().sort_index()
            labels = [target_names[i] for i in counts.index] if target_names is not None else counts.index
            fig = px.bar(x=labels, y=counts.values, title="Class balance", labels={"x": "class", "y": "count"})
            theme.style_fig(fig, 320); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            fig = px.histogram(y, nbins=30, title="Target distribution (disease progression score)")
            theme.style_fig(fig, 320); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            top_feats = feature_names[:6]
            fig = px.box(df, y=top_feats, title="Feature spread (first 6 features)")
            theme.style_fig(fig, 360)
            st.plotly_chart(fig, width='stretch')
        with c2:
            corr = X[feature_names[:10]].corr()
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature correlation (first 10)")
            theme.style_fig(fig, 360)
            st.plotly_chart(fig, width='stretch')

        st.dataframe(df.head(10), width='stretch')

    # 3. Data Preparation ------------------------------------------------------------
    with tabs[2]:
        st.subheader("Split & scaling strategy")
        strat_note = "**stratified** by class (preserving class ratios in both splits)" if cfg["type"] == "classification" else "a plain random split (regression target has no classes to stratify)"
        st.markdown(
            f"""
1. **75/25 train/test split**, {strat_note}.
2. Every candidate model is wrapped in a `Pipeline([StandardScaler, model])` —
   scaling happens **inside** the pipeline, so `RandomizedSearchCV`'s
   cross-validation refits the scaler on each fold's training data only.
   Fitting the scaler once on the full dataset before splitting would leak
   test-set statistics into training — a classic, easy-to-miss mistake this
   app avoids by construction.
3. **Held-out test set is touched exactly once** — after hyperparameter
   search picks each model's best configuration via cross-validation, the
   test set is used only to report the final leaderboard numbers, never to
   choose between models.
            """
        )
        st.code(
            "pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])\n"
            "search = RandomizedSearchCV(pipe, param_distributions, cv=5, scoring='roc_auc')\n"
            "search.fit(X_train, y_train)  # scaler refit per fold, never sees X_test",
            language="python",
        )

    # 4. Modeling -----------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Leaderboard")
        lb, fitted, split, is_clf, target_names, feature_names = run_tournament(task_name, n_iter)
        score_col = "CV score (ROC-AUC)" if is_clf else "CV score (R2)"
        st.dataframe(
            lb.drop(columns="Best params").style.format({
                score_col: "{:.3f}", "Tuning time (s)": "{:.2f}",
                **({"Test accuracy": "{:.3f}", "Test F1 (weighted)": "{:.3f}"} if is_clf else {"Test R2": "{:.3f}", "Test MAE": "{:.2f}"}),
            }).background_gradient(subset=[score_col], cmap="Greens"),
            width='stretch',
        )
        with st.expander("Best hyperparameters found per model"):
            for _, row in lb.iterrows():
                st.caption(f"**{row['Model']}**: `{row['Best params']}`")

        fig = px.bar(lb, x="Model", y="Tuning time (s)", title="Search cost per model (wall-clock seconds)")
        theme.style_fig(fig, 320); fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, width='stretch')

    # 5. Evaluation -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Best model — detailed evaluation")
        lb, fitted, (X_train, X_test, y_train, y_test), is_clf, target_names, feature_names = run_tournament(task_name, n_iter)
        best_name = lb.iloc[0]["Model"]
        best_pipe = fitted[best_name]
        st.success(f"Best model: **{best_name}**", icon="🏆")

        if is_clf:
            pred = best_pipe.predict(X_test)
            cm = confusion_matrix(y_test, pred)
            labels = list(target_names) if target_names is not None else sorted(y.unique())
            fig = px.imshow(cm, text_auto=True, x=[f"Pred: {l}" for l in labels], y=[f"True: {l}" for l in labels],
                             color_continuous_scale="Tealgrn", title="Confusion matrix")
            theme.style_fig(fig, 400)
            st.plotly_chart(fig, width='stretch')

            if len(np.unique(y)) == 2 and hasattr(best_pipe, "predict_proba"):
                proba = best_pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                auc = roc_auc_score(y_test, proba)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})", line=dict(color="#6ee7f2", width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#93a1c2", dash="dash"), name="random"))
                fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                theme.style_fig(fig, 380)
                st.plotly_chart(fig, width='stretch')
        else:
            pred = best_pipe.predict(X_test)
            fig = px.scatter(x=y_test, y=pred, labels={"x": "Actual", "y": "Predicted"}, title=f"Predicted vs actual — {best_name}")
            lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#fb7185", dash="dash"), name="perfect"))
            theme.style_fig(fig, 400)
            st.plotly_chart(fig, width='stretch')

        if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
            imp = pd.Series(best_pipe.named_steps["model"].feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
            fig = px.bar(imp[::-1], orientation="h", title=f"Feature importance — {best_name}")
            theme.style_fig(fig, 380); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

    # 6. Deployment -----------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Score a new example with the AutoML winner")
        lb, fitted, split, is_clf, target_names, feature_names = run_tournament(task_name, n_iter)
        best_name = lb.iloc[0]["Model"]
        best_pipe = fitted[best_name]
        st.caption(f"Using **{best_name}** (top of the leaderboard).")

        top_feats = feature_names[:6]
        inputs = X[feature_names].median().to_dict()
        cols = st.columns(3)
        for i, f in enumerate(top_feats):
            lo, hi = float(X[f].quantile(0.01)), float(X[f].quantile(0.99))
            inputs[f] = cols[i % 3].slider(f, lo, hi, float(X[f].median()))

        if st.button("🤖 Predict", type="primary"):
            x = pd.DataFrame([inputs])[feature_names]
            if is_clf:
                pred = best_pipe.predict(x)[0]
                label = target_names[pred] if target_names is not None else pred
                st.metric("Predicted class", str(label))
                if hasattr(best_pipe, "predict_proba"):
                    proba = best_pipe.predict_proba(x)[0]
                    proba_df = pd.DataFrame({"class": [target_names[i] if target_names is not None else i for i in range(len(proba))], "probability": proba})
                    fig = px.bar(proba_df, x="class", y="probability", title="Class probabilities")
                    theme.style_fig(fig, 300); fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')
            else:
                pred = float(best_pipe.predict(x)[0])
                st.metric("Predicted disease progression score", f"{pred:.1f}")

    theme.footer("06 · AutoML Model Tournament")


if __name__ == "__main__":
    main()
