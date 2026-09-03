"""Market Basket / Associative Pattern Mining
================================================
CRISP-DM project mining "frequently bought together" rules from ~4,000
real UK online-retail invoices (UCI Online Retail dataset) using Apriori
and FP-Growth (mlxtend).

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
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "data" / "basket_transactions.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])


@st.cache_data
def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    baskets = df.groupby("InvoiceNo")["Description"].apply(list)
    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets)
    return pd.DataFrame(arr, columns=te.columns_, index=baskets.index)


@st.cache_data(show_spinner=True)
def mine_rules(basket_df: pd.DataFrame, min_support: float, min_confidence: float, algo: str):
    t0 = time.perf_counter()
    fn = apriori if algo == "Apriori" else fpgrowth
    freq = fn(basket_df, min_support=min_support, use_colnames=True)
    elapsed = time.perf_counter() - t0
    if freq.empty:
        return freq, pd.DataFrame(), elapsed
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence, num_itemsets=len(freq))
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return freq, rules, elapsed


def main():
    theme.apply("🛒", "Market Basket Pattern Mining")
    theme.hero(
        "🛒", "Market Basket / Associative Pattern Mining",
        "CRISP-DM association-rule mining on 4,000 real UK online-retail invoices "
        "(UCI 'Online Retail') — Apriori & FP-Growth, support/confidence/lift.",
        ["Association Rules", "Apriori · FP-Growth", "mlxtend", "Streamlit"],
    )

    df = load_data()
    basket_df = build_basket_matrix(df)

    with st.sidebar:
        st.subheader("📁 Dataset")
        st.metric("Invoices (baskets)", f"{basket_df.shape[0]:,}")
        st.metric("Distinct products", f"{basket_df.shape[1]:,}")
        st.caption(
            "Source: UCI 'Online Retail' dataset — a UK-based online gift retailer, "
            "Dec 2010-Dec 2011. Restricted to UK invoices and the 120 most frequent SKUs "
            "for a tractable, still-realistic rule-mining demo."
        )
        st.divider()
        st.subheader("⚙️ Mining parameters")
        min_support = st.slider("Minimum support", 0.005, 0.10, 0.02, 0.005,
                                 help="Fraction of baskets that must contain the itemset.")
        min_confidence = st.slider("Minimum confidence", 0.05, 0.9, 0.25, 0.05,
                                    help="P(consequent | antecedent) threshold.")
        algo = st.radio("Algorithm", ["FP-Growth", "Apriori"], help="Both find the same frequent itemsets; FP-Growth avoids Apriori's candidate-generation pass and is typically faster on dense baskets.")

    freq, rules, elapsed = mine_rules(basket_df, min_support, min_confidence, algo)

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("Business objective")
        st.markdown(
            """
Merchandising wants **"customers who bought X also bought Y"** signals to drive
cross-sell placement, bundle promotions, and homepage recommendations — the classic
market-basket-analysis use case (famously popularized by the "beer and diapers" story).

**Success criteria (data science):** discover rules with high **lift** (>1, ideally
≫1 — meaning the items co-occur far more than chance) at workable support/confidence.

**Success criteria (business):** rules must be *frequent enough to matter* (appear in
enough real baskets) and *confident enough to act on* (a shopper who buys the
antecedent really does usually buy the consequent).
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Task</div>'
                     '<div class="ds-metric-value">Association Rules</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="ds-card"><div class="ds-metric-label">Algorithms</div>'
                     '<div class="ds-metric-value">Apriori · FP-Growth</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Primary metric</div>'
                     '<div class="ds-metric-value">Lift</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding ---------------------------------------------------------
    with tabs[1]:
        st.subheader("Exploratory data analysis")
        basket_sizes = df.groupby("InvoiceNo")["Description"].nunique()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(basket_sizes, nbins=40, title="Basket size distribution (items/invoice)")
            theme.style_fig(fig, 340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with c2:
            top_products = df["Description"].value_counts().head(15).sort_values()
            fig = px.bar(top_products, orientation="h", title="Top 15 products by invoice appearances")
            theme.style_fig(fig, 340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        daily = df.set_index("InvoiceDate").resample("D")["InvoiceNo"].nunique()
        fig = px.line(daily, title="Invoices per day (UK, top-120-product subset)")
        theme.style_fig(fig, 300)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

        st.dataframe(df.head(15), width='stretch')

    # 3. Data Preparation ------------------------------------------------------------
    with tabs[2]:
        st.subheader("Transaction encoding")
        st.markdown(
            """
Raw data is one row per **invoice line item** (`InvoiceNo`, `Description`, `Quantity`, ...).
Association-rule mining needs a **one-hot basket matrix**: one row per invoice, one
boolean column per product, `True` if that invoice contains that product. Built here
with `mlxtend.preprocessing.TransactionEncoder`.
            """
        )
        st.code(
            "baskets = df.groupby('InvoiceNo')['Description'].apply(list)\n"
            "te = TransactionEncoder()\n"
            "basket_matrix = pd.DataFrame(te.fit_transform(baskets), columns=te.columns_)",
            language="python",
        )
        st.dataframe(basket_df.iloc[:8, :10], width='stretch')
        density = basket_df.values.mean()
        st.metric("Matrix density (avg. items present per product column)", f"{density:.2%}")

    # 4. Modeling -----------------------------------------------------------------------
    with tabs[3]:
        st.subheader(f"Frequent itemsets — {algo}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Frequent itemsets found", f"{len(freq):,}")
        m2.metric("Rules above thresholds", f"{len(rules):,}")
        m3.metric("Mining time", f"{elapsed*1000:.0f} ms")

        if freq.empty:
            st.warning("No itemsets meet this support threshold — lower minimum support in the sidebar.")
        else:
            freq_sorted = freq.assign(size=freq["itemsets"].apply(len)).sort_values("support", ascending=False)
            freq_sorted["items"] = freq_sorted["itemsets"].apply(lambda s: ", ".join(sorted(s)))
            fig = px.bar(freq_sorted.head(15).sort_values("support"), x="support", y="items", orientation="h",
                         title="Top 15 most frequent itemsets", color="size", color_continuous_scale="Tealgrn")
            theme.style_fig(fig, 420)
            st.plotly_chart(fig, width='stretch')

            with st.expander("Compare Apriori vs FP-Growth timing at these thresholds"):
                _, _, t_ap = mine_rules(basket_df, min_support, min_confidence, "Apriori")
                _, _, t_fp = mine_rules(basket_df, min_support, min_confidence, "FP-Growth")
                comp = pd.DataFrame({"algorithm": ["Apriori", "FP-Growth"], "seconds": [t_ap, t_fp]})
                fig = px.bar(comp, x="algorithm", y="seconds", color="algorithm", title="Wall-clock time")
                theme.style_fig(fig, 300)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')

    # 5. Evaluation -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Rule quality")
        if rules.empty:
            st.warning("No rules meet the current support/confidence thresholds — relax them in the sidebar.")
        else:
            fig = px.scatter(
                rules, x="support", y="confidence", size="lift", color="lift",
                hover_data=["antecedents_str", "consequents_str"], color_continuous_scale="Tealgrn",
                title="Rules: support vs confidence (bubble size/color = lift)",
            )
            theme.style_fig(fig, 420)
            st.plotly_chart(fig, width='stretch')

            top_rules = rules.head(20)[["antecedents_str", "consequents_str", "support", "confidence", "lift"]]
            top_rules.columns = ["If basket has...", "...then also buys", "Support", "Confidence", "Lift"]
            st.markdown("#### Top 20 rules by lift")
            st.dataframe(
                top_rules.style.format({"Support": "{:.3f}", "Confidence": "{:.2f}", "Lift": "{:.2f}"})
                .background_gradient(subset=["Lift"], cmap="Greens"),
                width='stretch',
            )

    # 6. Deployment -----------------------------------------------------------------------
    with tabs[5]:
        st.subheader("🛍️ \"Frequently bought together\" recommender")
        if rules.empty:
            st.warning("No rules available at current thresholds — relax them in the sidebar to enable recommendations.")
        else:
            all_products = sorted(basket_df.columns)
            cart = st.multiselect("Items currently in cart", all_products,
                                   default=[all_products[0]] if all_products else [])

            if cart:
                cart_set = set(cart)
                fires = rules["antecedents"].apply(lambda s: s.issubset(cart_set))
                not_owned = rules["consequents"].apply(lambda s: not s.issubset(cart_set))
                matches = rules[fires & not_owned].sort_values("lift", ascending=False)

                if matches.empty:
                    st.info("No confident rule fires for this exact cart at current thresholds — try lowering "
                             "minimum support/confidence in the sidebar, or add a top product to the cart.", icon="🤷")
                else:
                    st.success(f"Found {len(matches)} candidate recommendation rule(s).", icon="🛒")
                    recs = {}
                    for _, r in matches.iterrows():
                        for item in r["consequents"]:
                            if item not in recs or r["lift"] > recs[item]["lift"]:
                                recs[item] = {"lift": r["lift"], "confidence": r["confidence"], "via": r["antecedents_str"]}
                    rec_df = pd.DataFrame(recs).T.sort_values("lift", ascending=False).head(8)
                    rec_df.index.name = "Recommended product"
                    st.dataframe(
                        rec_df.rename(columns={"lift": "Lift", "confidence": "Confidence", "via": "Because you have"})
                        .style.format({"Lift": "{:.2f}", "Confidence": "{:.2f}"}),
                        width='stretch',
                    )

                    fig = go.Figure()
                    for item in cart:
                        fig.add_trace(go.Scatter(x=[0], y=[item], mode="markers+text", text=[item], textposition="middle left",
                                                  marker=dict(size=16, color="#6ee7f2"), showlegend=False))
                    for item in rec_df.index[:6]:
                        fig.add_trace(go.Scatter(x=[1], y=[item], mode="markers+text", text=[item], textposition="middle right",
                                                  marker=dict(size=16, color="#fbbf24"), showlegend=False))
                    for item in rec_df.index[:6]:
                        for c in cart:
                            fig.add_trace(go.Scatter(x=[0, 1], y=[c, item], mode="lines",
                                                      line=dict(color="#243050", width=1), showlegend=False))
                    fig.update_layout(xaxis=dict(visible=False, range=[-0.5, 1.5]), yaxis=dict(visible=False),
                                       title="Cart → Recommendations")
                    theme.style_fig(fig, 420)
                    st.plotly_chart(fig, width='stretch')
            else:
                st.caption("Add one or more products to the cart to see recommendations.")

    theme.footer("03 · Market Basket Pattern Mining")


if __name__ == "__main__":
    main()
