"""Enterprise Data Science Audit
===================================
A genuine static-analysis auditor — not a mockup. This app reads the
**actual source files** of every other project in this very portfolio
(`../01_.../app.py` through `../10_.../app.py`, plus their READMEs) at
runtime, and scores each one against a real reproducibility/leakage/
documentation rubric using plain regex/substring checks on real code. Every
number in this app's Evaluation tab is computed live from the current state
of this repository, not hardcoded.

Run:
    streamlit run app.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THIS_PROJECT = Path(__file__).resolve().parent.name

# Projects that center on a predictive pipeline (leakage/metric rigor checks
# apply in full) vs. tools/teaching aids (documentation/reproducibility
# checks still apply; a missing train/test split isn't a defect there).
PREDICTIVE_PROJECTS = {
    "01_nyc_taxi_trip_duration", "04_fraud_anomaly_detection",
    "05_time_series_forecasting", "06_automl_model_tournament",
    "10_crispdm_masters_curriculum",
}
# Unsupervised projects have no single held-out target, so a classic
# train_test_split isn't the relevant leakage check for them — scoring them
# against that rubric would be a categorization bug, not a fair finding.
UNSUPERVISED_PROJECTS = {"02_customer_segmentation_clustering", "03_market_basket_mining"}

METRIC_KEYWORDS = [
    "roc_auc", "average_precision", "f1_score", "precision_recall_curve", "mean_absolute_error",
    "root_mean_squared_error", "r2_score", "mape", "silhouette_score", "confusion_matrix", "adjusted_rand_score",
]
MODEL_KEYWORDS = [
    "LogisticRegression", "RandomForest", "GradientBoosting", "KMeans", "DBSCAN", "AgglomerativeClustering",
    "SARIMAX", "ExponentialSmoothing", "KNeighbors", "SVC(", "SVR(", "GaussianNB", "IsolationForest",
    "DecisionTree", "Ridge(", "NanoGPT", "MultinomialNB",
]


@st.cache_data
def discover_projects() -> list[str]:
    projects = []
    for p in sorted(ROOT.iterdir()):
        if p.is_dir() and re.match(r"^\d\d_", p.name) and (p / "app.py").exists():
            projects.append(p.name)
    return projects


@st.cache_data
def audit_project(name: str) -> dict:
    proj_dir = ROOT / name
    app_code = (proj_dir / "app.py").read_text(errors="ignore")
    readme = (proj_dir / "README.md").read_text(errors="ignore") if (proj_dir / "README.md").exists() else ""
    has_prompts = (proj_dir / "PROMPTS.md").exists()
    is_predictive = name in PREDICTIVE_PROJECTS
    is_unsupervised = name in UNSUPERVISED_PROJECTS
    project_type = "Predictive pipeline" if is_predictive else ("Unsupervised pipeline" if is_unsupervised else "Tool / teaching")

    findings = []

    # --- Reproducibility ---
    has_cache = "@st.cache_data" in app_code or "@st.cache_resource" in app_code
    n_random_state = len(re.findall(r"random_state\s*=\s*\d+", app_code))
    has_docstring = app_code.lstrip().startswith('"""')
    repro_score = 40 * has_cache + 30 * (n_random_state > 0) + 30 * has_docstring
    if not has_cache:
        findings.append("No `@st.cache_data`/`@st.cache_resource` found — app may recompute expensive work on every rerun.")
    if n_random_state == 0:
        findings.append("No pinned `random_state=` found — results may not be exactly reproducible run to run.")

    # --- Leakage risk (predictive projects only) ---
    # A held-out split can be sklearn's train_test_split, or (for time series)
    # an explicit chronological/walk-forward split — both count as "has a split".
    has_sklearn_split = "train_test_split" in app_code
    has_time_split = bool(re.search(r"backtest|walk_forward", app_code, re.I))
    has_split = has_sklearn_split or has_time_split
    has_stratify = "stratify=" in app_code
    has_pipeline = "Pipeline(" in app_code or "Pipeline([" in app_code
    # A scaler fit explicitly on a train-only variable (not wrapped in a
    # Pipeline) is also leakage-safe — recognize that pattern too, not just
    # the Pipeline idiom.
    has_train_only_fit = bool(re.search(r"scaler\s*=\s*\w+\(\s*\)\.fit\(\s*X_train", app_code))
    has_scaler = "StandardScaler" in app_code
    scaler_safe = has_pipeline or has_train_only_fit or not has_scaler
    if is_predictive:
        leakage_score = 40 * has_split + 20 * has_stratify + 40 * scaler_safe
        if not has_split:
            findings.append("No `train_test_split` (or explicit chronological/backtest split) found — verify held-out evaluation exists.")
        if has_scaler and not scaler_safe:
            findings.append("A scaler is used but not obviously fit on train-only data or wrapped in a `Pipeline` — verify it isn't fit on the full dataset.")
    else:
        leakage_score = None  # not applicable to unsupervised/tool projects

    # --- Evaluation rigor (predictive + unsupervised pipelines; not applicable to pure tools) ---
    is_pipeline = is_predictive or is_unsupervised
    n_metrics = sum(kw in app_code for kw in METRIC_KEYWORDS)
    accuracy_only = ("accuracy_score" in app_code) and n_metrics == 0
    if is_pipeline:
        eval_score = min(100, n_metrics * 34)
        if accuracy_only:
            findings.append("Only `accuracy_score` detected as an evaluation metric — verify this isn't an imbalanced problem.")
        if n_metrics == 0 and not is_unsupervised:
            findings.append("No recognized evaluation metric detected at all — verify results are being measured, not just displayed.")
    else:
        eval_score = None

    # --- Model rigor: how many distinct model/algorithm families are compared ---
    n_models = sum(kw in app_code for kw in MODEL_KEYWORDS)
    if is_pipeline and n_models <= 1:
        findings.append(f"Only {n_models} model family detected — verify a real comparison/tournament exists, not a single hardcoded model.")

    # --- Documentation completeness ---
    word_count = len(readme.split())
    n_screenshots = len(re.findall(r"\.png\)", readme))
    has_limitations = bool(re.search(r"honest limitations|## limitations", readme, re.I))
    has_how_built = bool(re.search(r"how it'?s built", readme, re.I))
    has_root_link = "../README.md" in readme
    doc_score = (
        25 * (word_count > 300) + 25 * (n_screenshots >= 4) +
        25 * has_limitations + 15 * has_how_built + 10 * has_root_link
    )
    if word_count < 300:
        findings.append(f"README is short ({word_count} words) — may be under-documented.")
    if n_screenshots < 4:
        findings.append(f"Only {n_screenshots} screenshot(s) linked in README — expected a tour of all 6 CRISP-DM tabs.")
    if not has_limitations:
        findings.append("No 'Honest limitations' section found — verify the README discloses known weaknesses.")
    if not has_prompts:
        findings.append("No PROMPTS.md found for this project.")

    dims = {"Reproducibility": repro_score, "Documentation": doc_score}
    if is_predictive:
        dims["Leakage risk mitigation"] = leakage_score
    if is_pipeline:
        dims["Evaluation rigor"] = eval_score
    overall = sum(dims.values()) / len(dims)

    return {
        "project": name, "type": project_type,
        "loc": len(app_code.splitlines()), "readme_words": word_count, "n_screenshots": n_screenshots,
        "has_prompts_md": has_prompts, "n_models_detected": n_models, "n_metrics_detected": n_metrics,
        "dims": dims, "overall": overall, "findings": findings,
    }


def grade_letter(score: float) -> str:
    if score >= 95: return "A+"
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    return "D"


def audit_pasted_code(code: str) -> dict:
    """Lighter version of audit_project() for arbitrary pasted code (Deployment tab)."""
    has_split = "train_test_split" in code
    has_stratify = "stratify=" in code
    has_pipeline = "Pipeline(" in code
    has_scaler = "StandardScaler" in code or "MinMaxScaler" in code
    has_cache = "@st.cache_data" in code or "@st.cache_resource" in code or "@lru_cache" in code
    n_random_state = len(re.findall(r"random_state\s*=\s*\d+", code))
    n_metrics = sum(kw in code for kw in METRIC_KEYWORDS + ["accuracy_score"])
    return {
        "Uses train_test_split": has_split,
        "Uses stratify=": has_stratify,
        "Scaler inside a Pipeline (leakage-safe)": has_pipeline if has_scaler else "n/a (no scaler detected)",
        "Has caching": has_cache,
        "Pins random_state": n_random_state > 0,
        "Evaluation metrics detected": n_metrics,
    }


def main():
    theme.apply("🕵️", "Enterprise Data Science Audit")
    theme.hero(
        "🕵️", "Enterprise Data Science Audit",
        "A real static-analysis auditor that reads this portfolio's own source files at "
        "runtime and scores every project for reproducibility, leakage-risk mitigation, "
        "evaluation rigor, and documentation completeness — no hardcoded results.",
        ["Static analysis", "Self-auditing portfolio", "Governance", "Streamlit"],
    )

    projects = [p for p in discover_projects() if p != THIS_PROJECT]
    audits = {p: audit_project(p) for p in projects}
    overall_scores = pd.Series({p: a["overall"] for p, a in audits.items()})
    portfolio_grade = grade_letter(overall_scores.mean())

    with st.sidebar:
        st.subheader("📁 Audit target")
        st.metric("Projects scanned", len(projects))
        st.metric("Portfolio average score", f"{overall_scores.mean():.1f}/100")
        st.metric("Portfolio grade", portfolio_grade)
        st.caption("This app excludes itself from the scan — auditing the auditor is left as an exercise.")

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Why audit a data science portfolio")
        st.markdown(
            """
As a portfolio of models grows, so does the risk of an easy-to-miss mistake
slipping into production: a scaler fit before the train/test split, an
evaluation metric that flatters an imbalanced model, a model nobody can
reproduce because a random seed was never pinned. A **governance audit**
catches these systematically instead of relying on every reviewer
re-deriving the checklist from memory.

**This app is a real auditor, not a scripted demo** — it reads every other
project's actual `app.py` and `README.md` from disk each time you load it,
and every score below is computed live from what it finds.

**Primary output:** a per-project scorecard across 4 dimensions
(Reproducibility, Documentation, Leakage-risk mitigation, Evaluation
rigor — the latter two only where applicable), plus a portfolio-wide grade.
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Method</div>'
                     '<div class="ds-metric-value">Static source analysis</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="ds-card"><div class="ds-metric-label">Projects audited</div>'
                     f'<div class="ds-metric-value">{len(projects)}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="ds-card"><div class="ds-metric-label">Portfolio grade</div>'
                     f'<div class="ds-metric-value">{portfolio_grade}</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding ---------------------------------------------------------
    with tabs[1]:
        st.subheader("The 'data' here is the portfolio itself")
        overview = pd.DataFrame([{
            "Project": p, "Type": a["type"], "app.py LOC": a["loc"], "README words": a["readme_words"],
            "Screenshots": a["n_screenshots"], "Has PROMPTS.md": a["has_prompts_md"],
            "Models/algorithms detected": a["n_models_detected"],
        } for p, a in audits.items()])
        st.dataframe(overview, width='stretch', hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(overview, x="Project", y="app.py LOC", title="Lines of code per app.py")
            theme.style_fig(fig, 360); fig.update_layout(showlegend=False, xaxis_tickangle=-40)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.bar(overview, x="Project", y="README words", title="README word count per project")
            theme.style_fig(fig, 360); fig.update_layout(showlegend=False, xaxis_tickangle=-40)
            st.plotly_chart(fig, width='stretch')

    # 3. Data Preparation → Audit methodology, fully disclosed --------------------------
    with tabs[2]:
        st.subheader("Audit methodology — fully disclosed")
        st.markdown(
            """
A trustworthy audit discloses exactly how it scores. Every check below is a
plain regex/substring search over each project's real `app.py` and
`README.md` — no AI judgment call, no hidden heuristic.
            """
        )
        methodology = pd.DataFrame([
            {"Dimension": "Reproducibility", "Checks": "@st.cache_data/@st.cache_resource present · random_state= pinned · module docstring present", "Applies to": "All projects"},
            {"Dimension": "Documentation", "Checks": "README > 300 words · ≥4 screenshots linked · 'Honest limitations' section · 'How it's built' section · links back to root README · has PROMPTS.md", "Applies to": "All projects"},
            {"Dimension": "Leakage risk mitigation", "Checks": "train_test_split (or an explicit chronological/backtest split) present · stratify= used · scaler is inside a Pipeline or explicitly fit on a train-only variable", "Applies to": "Predictive-pipeline projects only"},
            {"Dimension": "Evaluation rigor", "Checks": "count of non-accuracy metrics detected (ROC-AUC, F1, MAE, RMSE, R², silhouette, ARI, ...) · flags accuracy-only evaluation · flags a single hardcoded model with no comparison", "Applies to": "Predictive + unsupervised pipelines"},
        ])
        st.dataframe(methodology, width='stretch', hide_index=True)
        st.caption(
            "Three project types, scored differently on purpose: **predictive pipelines** (01, 04, 05, 06, 10) "
            "get all four dimensions; **unsupervised pipelines** (02, 03) skip leakage-risk since there's no "
            "single held-out target to leak into; **tools/teaching** (07, 08, 09) skip both leakage and "
            "evaluation-rigor entirely — they were never built to predict a held-out target, and scoring them "
            "as if they should would be exactly the kind of unfair, context-blind audit this app is trying not "
            "to be. This categorization is itself a hardcoded judgment call, disclosed here rather than hidden."
        )

    # 4. Modeling → run the audit -----------------------------------------------------------
    with tabs[3]:
        st.subheader("Run the audit")
        score_df = pd.DataFrame({p: a["dims"] for p, a in audits.items()}).T
        score_df["Overall"] = overall_scores
        score_df["Grade"] = score_df["Overall"].apply(grade_letter)
        st.dataframe(
            score_df.style.format("{:.0f}", subset=[c for c in score_df.columns if c != "Grade"])
            .background_gradient(subset=["Overall"], cmap="Greens"),
            width='stretch',
        )

        fig = px.bar(score_df.reset_index(), x="index", y="Overall", color="Grade", title="Overall score by project")
        theme.style_fig(fig, 380); fig.update_layout(xaxis_title="", xaxis_tickangle=-40)
        st.plotly_chart(fig, width='stretch')

    # 5. Evaluation → scorecards + flagged findings ------------------------------------------
    with tabs[4]:
        st.subheader("Per-project scorecards & findings")
        project = st.selectbox("Inspect a project", projects)
        a = audits[project]
        m1, m2 = st.columns(2)
        m1.metric("Overall score", f"{a['overall']:.0f}/100")
        m2.metric("Grade", grade_letter(a["overall"]))

        fig = go.Figure()
        dims = a["dims"]
        fig.add_trace(go.Scatterpolar(r=list(dims.values()) + [list(dims.values())[0]],
                                        theta=list(dims.keys()) + [list(dims.keys())[0]], fill="toself"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title=f"{project} — dimension radar")
        theme.style_fig(fig, 400)
        st.plotly_chart(fig, width='stretch')

        if a["findings"]:
            st.markdown("#### 🚩 Findings")
            for f in a["findings"]:
                st.warning(f, icon="🚩")
        else:
            st.success("No findings — every automated check passed.", icon="✅")

    # 6. Deployment → audit arbitrary code -----------------------------------------------------
    with tabs[5]:
        st.subheader("Audit your own code")
        st.caption("Paste any Python/Streamlit source and run the same reproducibility checks live.")
        sample = (ROOT / projects[0] / "app.py").read_text(errors="ignore")[:800] if projects else ""
        code = st.text_area("Paste app.py code", value=sample, height=240)
        if st.button("🕵️ Audit this code", type="primary"):
            result = audit_pasted_code(code)
            res_df = pd.DataFrame(result.items(), columns=["Check", "Result"])
            st.dataframe(res_df, width='stretch', hide_index=True)

    theme.footer("11 · Enterprise Data Science Audit")


if __name__ == "__main__":
    main()
