"""Data Science Skills Lab
=============================
A catalog of 10 individual, foundational data science skills — statistical
testing, outlier detection, feature engineering, RFM scoring, cohort
retention, dimensionality reduction, business metrics — each demonstrated
**live** on real data already used elsewhere in this portfolio (Mall
Customers, the Online Retail RFM table, and its invoice-level transactions).

Where projects 01-08 each go deep on *one* technique end-to-end, this
project goes *wide*: a browsable toolkit, the kind of skill inventory a
data scientist builds up over a career and reaches for interview-prep or a
new dataset.

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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_data():
    mall = pd.read_csv(DATA_DIR / "Mall_Customers.csv")
    rfm = pd.read_csv(DATA_DIR / "retail_rfm.csv")
    baskets = pd.read_csv(DATA_DIR / "basket_transactions.csv", parse_dates=["InvoiceDate"])
    return mall, rfm, baskets


# ---------------------------------------------------------------------------
# Each skill is a self-contained function: (mall, rfm, baskets) -> renders itself
# ---------------------------------------------------------------------------
def skill_descriptive_stats(mall, rfm, baskets):
    st.markdown("Summarize the shape of a variable beyond just its mean — spread, skew, and tails.")
    col = st.selectbox("Variable", ["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    s = mall[col]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{s.mean():.1f}")
    c2.metric("Median", f"{s.median():.1f}")
    c3.metric("Std dev", f"{s.std():.1f}")
    c4.metric("Skewness", f"{s.skew():.2f}", help=">0 = right-tailed, <0 = left-tailed, ~0 = symmetric")
    fig = px.histogram(mall, x=col, marginal="box", nbins=25, title=f"Distribution of {col}")
    theme.style_fig(fig, 380)
    st.plotly_chart(fig, width='stretch')


def skill_correlation_test(mall, rfm, baskets):
    st.markdown("Quantify *and test* whether two variables move together — a correlation alone can't tell you if it's statistically meaningful.")
    x_col = st.selectbox("X", ["Age", "Annual Income (k$)"], key="corr_x")
    y_col = st.selectbox("Y", ["Spending Score (1-100)"], key="corr_y")
    r, p = stats.pearsonr(mall[x_col], mall[y_col])
    c1, c2 = st.columns(2)
    c1.metric("Pearson r", f"{r:.3f}")
    c2.metric("p-value", f"{p:.4f}", help="< 0.05 conventionally means the correlation is unlikely to be chance")
    verdict = "statistically significant" if p < 0.05 else "not statistically significant"
    st.info(f"At α=0.05, this correlation is **{verdict}**.", icon="🧪")
    fig = px.scatter(mall, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs {y_col}")
    theme.style_fig(fig, 400)
    st.plotly_chart(fig, width='stretch')


def skill_outlier_detection(mall, rfm, baskets):
    st.markdown("Two classic univariate outlier rules, compared side by side.")
    col = st.selectbox("Variable", ["Annual Income (k$)", "Spending Score (1-100)", "Age"], key="outlier_col")
    s = mall[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_outliers = mall[(s < lo) | (s > hi)]
    z = (s - s.mean()) / s.std()
    z_outliers = mall[z.abs() > 3]

    c1, c2 = st.columns(2)
    c1.metric("IQR method flags", len(iqr_outliers), help=f"Outside [{lo:.1f}, {hi:.1f}]")
    c2.metric("Z-score method flags (|z|>3)", len(z_outliers))

    fig = px.box(mall, y=col, points="all", title=f"{col} — Tukey box plot (whiskers = 1.5×IQR)")
    theme.style_fig(fig, 400)
    st.plotly_chart(fig, width='stretch')


def skill_ab_test(mall, rfm, baskets):
    st.markdown("Compare a metric across two groups — the core mechanic behind every A/B test readout.")
    metric = st.selectbox("Metric", ["Spending Score (1-100)", "Annual Income (k$)"], key="ab_metric")
    g1 = mall[mall["Genre"] == "Male"][metric]
    g2 = mall[mall["Genre"] == "Female"][metric]
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Male mean", f"{g1.mean():.1f}")
    c2.metric("Female mean", f"{g2.mean():.1f}")
    c3.metric("p-value (Welch's t-test)", f"{p_val:.3f}")
    verdict = "a statistically significant difference" if p_val < 0.05 else "no statistically significant difference"
    st.info(f"At α=0.05, there is **{verdict}** between groups.", icon="🧪")
    fig = px.violin(mall, x="Genre", y=metric, box=True, color="Genre", title=f"{metric} by group")
    theme.style_fig(fig, 380); fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')


def skill_chi_square(mall, rfm, baskets):
    st.markdown("Test whether two **categorical** variables are independent — e.g. does gender relate to income bracket?")
    df = mall.copy()
    df["income_bracket"] = pd.qcut(df["Annual Income (k$)"], 3, labels=["Low", "Mid", "High"])
    ct = pd.crosstab(df["Genre"], df["income_bracket"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    st.dataframe(ct, width='stretch')
    c1, c2 = st.columns(2)
    c1.metric("Chi-square statistic", f"{chi2:.2f}")
    c2.metric("p-value", f"{p:.3f}")
    verdict = "dependent (related)" if p < 0.05 else "independent (unrelated)"
    st.info(f"At α=0.05, gender and income bracket appear **{verdict}**.", icon="🧪")
    fig = px.bar(ct, barmode="group", title="Gender × Income bracket")
    theme.style_fig(fig, 340)
    st.plotly_chart(fig, width='stretch')


def skill_feature_engineering(mall, rfm, baskets):
    st.markdown("Raw columns rarely capture the signal a model needs directly — derive ratios and interactions instead.")
    df = mall.copy()
    df["income_per_age"] = df["Annual Income (k$)"] / df["Age"]
    df["spend_efficiency"] = df["Spending Score (1-100)"] / df["Annual Income (k$)"]
    st.code(
        "df['income_per_age'] = df['Annual Income (k$)'] / df['Age']\n"
        "df['spend_efficiency'] = df['Spending Score (1-100)'] / df['Annual Income (k$)']",
        language="python",
    )
    st.dataframe(df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "income_per_age", "spend_efficiency"]].head(10), width='stretch')
    fig = px.scatter(df, x="income_per_age", y="spend_efficiency", color="Genre", title="Two engineered features")
    theme.style_fig(fig, 380)
    st.plotly_chart(fig, width='stretch')


def skill_rfm_scoring(mall, rfm, baskets):
    st.markdown("Quantile-score customers 1-5 on Recency, Frequency, and Monetary value — a lighter-weight alternative to full clustering ([project 02](../02_customer_segmentation_clustering)) for a quick customer-value tier.")
    df = rfm.copy()
    df["R_score"] = pd.qcut(df["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_score"] = pd.qcut(df["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["RFM_score"] = df["R_score"] + df["F_score"] + df["M_score"]

    def tier(s):
        if s >= 13: return "Champion"
        if s >= 10: return "Loyal"
        if s >= 7: return "At Risk"
        return "Lost"
    df["tier"] = df["RFM_score"].apply(tier)

    st.dataframe(df[["CustomerID", "recency_days", "frequency", "monetary", "R_score", "F_score", "M_score", "RFM_score", "tier"]].head(10), width='stretch')
    counts = df["tier"].value_counts()
    fig = px.pie(counts, names=counts.index, values=counts.values, hole=0.5, title="Customer value tiers")
    theme.style_fig(fig, 380)
    st.plotly_chart(fig, width='stretch')


def skill_cohort_analysis(mall, rfm, baskets):
    st.markdown("Group customers by their **first purchase month**, then track what fraction are still buying in each month after — the classic retention/cohort heatmap.")
    df = baskets.copy()
    df["order_month"] = df["InvoiceDate"].dt.to_period("M")
    first_purchase = df.groupby("CustomerID")["order_month"].min().rename("cohort_month")
    df = df.join(first_purchase, on="CustomerID")
    df["period_number"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    cohort_data = df.groupby(["cohort_month", "period_number"])["CustomerID"].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period_number", values="CustomerID")
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0).round(3)
    retention.index = retention.index.astype(str)

    fig = px.imshow(retention.iloc[:, :6], text_auto=".0%", color_continuous_scale="Tealgrn",
                     labels=dict(x="Months since first purchase", y="Cohort (first purchase month)", color="Retention"),
                     title="Monthly cohort retention (first 6 months)")
    theme.style_fig(fig, 420)
    st.plotly_chart(fig, width='stretch')
    st.caption("Reads left→right, top→bottom: each row is a cohort of customers who first bought that month; "
               "each cell is the % of that cohort still buying N months later.")


def skill_pca(mall, rfm, baskets):
    st.markdown("Compress correlated numeric features into a few uncorrelated components that explain most of the variance.")
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = StandardScaler().fit_transform(mall[features])
    pca = PCA(n_components=3)
    comps = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(x=[f"PC{i+1}" for i in range(3)], y=var_ratio, title="Variance explained per component")
        theme.style_fig(fig, 340); fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width='stretch')
    with c2:
        proj = pd.DataFrame(comps[:, :2], columns=["PC1", "PC2"])
        proj["Genre"] = mall["Genre"].values
        fig = px.scatter(proj, x="PC1", y="PC2", color="Genre", title=f"2D projection ({var_ratio[:2].sum()*100:.0f}% variance kept)")
        theme.style_fig(fig, 340)
        st.plotly_chart(fig, width='stretch')
    loadings = pd.DataFrame(pca.components_[:2].T, columns=["PC1", "PC2"], index=features)
    st.dataframe(loadings.style.format("{:.2f}"), width='stretch')


def skill_business_metrics(mall, rfm, baskets):
    st.markdown("Translate raw customer data into the metrics a business stakeholder actually asks for.")
    total_customers = len(rfm)
    total_revenue = rfm["monetary"].sum()
    arpu = total_revenue / total_customers
    repeat_rate = (rfm["frequency"] > 1).mean()
    avg_clv_12mo = arpu * (rfm["frequency"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("ARPU (avg revenue/customer)", f"£{arpu:,.0f}")
    c3.metric("Repeat purchase rate", f"{repeat_rate*100:.1f}%")
    c4.metric("Rough CLV proxy", f"£{avg_clv_12mo:,.0f}")
    st.code(
        "arpu = total_revenue / total_customers\n"
        "repeat_rate = (frequency > 1).mean()\n"
        "clv_proxy = arpu * avg_frequency   # simplistic, no discounting/churn model",
        language="python",
    )
    fig = px.histogram(rfm, x="monetary", nbins=40, title="Revenue per customer distribution")
    theme.style_fig(fig, 340); fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')


SKILLS = {
    "Descriptive Statistics & Distribution Profiling": {"fn": skill_descriptive_stats, "category": "Statistics", "dataset": "Mall Customers"},
    "Correlation & Hypothesis Testing": {"fn": skill_correlation_test, "category": "Statistics", "dataset": "Mall Customers"},
    "Outlier Detection (IQR vs Z-score)": {"fn": skill_outlier_detection, "category": "Data Quality", "dataset": "Mall Customers"},
    "A/B / Group Comparison Testing": {"fn": skill_ab_test, "category": "Statistics", "dataset": "Mall Customers"},
    "Chi-Square Test of Independence": {"fn": skill_chi_square, "category": "Statistics", "dataset": "Mall Customers"},
    "Feature Engineering": {"fn": skill_feature_engineering, "category": "Data Prep", "dataset": "Mall Customers"},
    "RFM Customer Value Scoring": {"fn": skill_rfm_scoring, "category": "Segmentation", "dataset": "Online Retail RFM"},
    "Cohort Retention Analysis": {"fn": skill_cohort_analysis, "category": "Segmentation", "dataset": "Online Retail invoices"},
    "PCA / Dimensionality Reduction": {"fn": skill_pca, "category": "Modeling", "dataset": "Mall Customers"},
    "Business Metrics Calculator": {"fn": skill_business_metrics, "category": "Business", "dataset": "Online Retail RFM"},
}

RECOMMENDER = {
    "Are two groups meaningfully different?": ["A/B / Group Comparison Testing", "Chi-Square Test of Independence"],
    "Which data points look wrong or unusual?": ["Outlier Detection (IQR vs Z-score)"],
    "How do my variables relate to each other?": ["Correlation & Hypothesis Testing", "PCA / Dimensionality Reduction"],
    "Which customers are most valuable?": ["RFM Customer Value Scoring", "Business Metrics Calculator"],
    "Are customers coming back over time?": ["Cohort Retention Analysis"],
    "I need better inputs for a model": ["Feature Engineering", "PCA / Dimensionality Reduction"],
    "I just got a new dataset — where do I start?": ["Descriptive Statistics & Distribution Profiling", "Outlier Detection (IQR vs Z-score)"],
}


def main():
    theme.apply("🧰", "Data Science Skills Lab")
    theme.hero(
        "🧰", "Data Science Skills Lab",
        "A browsable catalog of 10 foundational data science skills — statistical testing, "
        "outlier detection, feature engineering, RFM scoring, cohort retention, PCA, business "
        "metrics — each run live on real data already used elsewhere in this portfolio.",
        ["10 skills", "scipy.stats", "Real data, no new downloads", "Streamlit"],
    )

    mall, rfm, baskets = load_data()

    with st.sidebar:
        st.subheader("📁 Datasets reused")
        st.caption("No new data sourced for this project — every skill below runs on datasets "
                   "already built for [project 02](../02_customer_segmentation_clustering) and "
                   "[project 03](../03_market_basket_mining).")
        st.metric("Mall Customers", f"{len(mall)} rows")
        st.metric("Online Retail RFM", f"{len(rfm)} rows")
        st.metric("Online Retail invoices", f"{len(baskets)} line items")

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Why a skills catalog")
        st.markdown(
            """
Projects 01-08 each go **deep** on one technique end-to-end. Real data
science work is often the opposite: a fast, wide toolkit you reach into for
whatever the next question demands. This project is that toolkit — 10
individually useful skills, each demonstrated live, each with the runnable
code shown alongside its output.
            """
        )
        cat_df = pd.DataFrame([{"Skill": k, "Category": v["category"], "Dataset": v["dataset"]} for k, v in SKILLS.items()])
        st.dataframe(cat_df, width='stretch', hide_index=True)

    # 2. Data Understanding ---------------------------------------------------------
    with tabs[1]:
        st.subheader("The three datasets behind every skill")
        st.markdown(
            """
* **Mall Customers** (200 rows) — Age, Annual Income, Spending Score, Genre.
  Small and clean, ideal for statistics/testing/EDA skills.
* **Online Retail RFM** (3,920 rows) — Recency/Frequency/Monetary per real
  customer, derived from UK invoices. Used for value-scoring and business
  metrics skills.
* **Online Retail invoices** (27,170 line items) — the same invoices at
  line-item grain with real dates, used for the cohort retention skill.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Mall Customers sample")
            st.dataframe(mall.head(6), width='stretch')
        with c2:
            st.caption("Online Retail RFM sample")
            st.dataframe(rfm.head(6), width='stretch')

    # 3. Data Preparation → skill catalog gallery --------------------------------------
    with tabs[2]:
        st.subheader("Skill catalog")
        cols = st.columns(2)
        for i, (name, meta) in enumerate(SKILLS.items()):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="ds-card"><b>{name}</b><br>'
                    f'<span class="ds-metric-label">{meta["category"]} · {meta["dataset"]}</span></div>',
                    unsafe_allow_html=True,
                )
                st.write("")

    # 4. Modeling → run a skill live -----------------------------------------------------
    with tabs[3]:
        st.subheader("Run a skill")
        skill_name = st.selectbox("Pick a skill", list(SKILLS.keys()))
        meta = SKILLS[skill_name]
        st.caption(f"Category: **{meta['category']}** · Dataset: **{meta['dataset']}**")
        meta["fn"](mall, rfm, baskets)

    # 5. Evaluation → skill inventory summary --------------------------------------------
    with tabs[4]:
        st.subheader("Skill inventory by category")
        cat_counts = pd.Series([v["category"] for v in SKILLS.values()]).value_counts()
        fig = px.bar(cat_counts, title="Skills per category")
        theme.style_fig(fig, 340); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            """
A well-rounded toolkit spans **statistics** (is this real or noise?),
**data quality** (can I trust this row?), **data prep** (is this the right
input shape?), **segmentation** (who matters most?), **modeling**
(what's the underlying structure?), and **business translation** (so what,
in dollars?). This catalog deliberately touches all six.
            """
        )

    # 6. Deployment → skill recommender --------------------------------------------------
    with tabs[5]:
        st.subheader("🧭 Which skill do I need?")
        question = st.selectbox("Describe your business question", list(RECOMMENDER.keys()))
        recs = RECOMMENDER[question]
        st.success(f"Recommended skill(s): **{', '.join(recs)}**", icon="🎯")
        st.caption("Head to the Modeling tab and pick one of the skills above to run it live.")

    theme.footer("09 · Data Science Skills Lab")


if __name__ == "__main__":
    main()
