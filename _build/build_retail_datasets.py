"""One-off script: clean the UCI 'Online Retail' transactional dataset
(a UK-based online gift retailer, Dec 2010-Dec 2011) and derive two
repo-friendly assets:

1. basket_transactions.csv - one row per invoice line item, restricted
   to the UK market and top-N most frequent products, for Apriori /
   FP-Growth association rule mining.
2. retail_rfm.csv - one row per customer with Recency/Frequency/Monetary
   aggregates, for an RFM-driven customer segmentation cross-check.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
xlsx_path = HERE / "online_retail.xlsx"
BASKET_OUT = HERE.parent / "03_market_basket_mining" / "data" / "basket_transactions.csv"
RFM_OUT = HERE.parent / "02_customer_segmentation_clustering" / "data" / "retail_rfm.csv"

if not xlsx_path.exists():
    print("Downloading Online Retail.xlsx from UCI ...")
    urllib.request.urlretrieve(SOURCE_URL, xlsx_path)

df = pd.read_excel(xlsx_path)

# --- Clean (CRISP-DM: Data Preparation) ---
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # drop cancellations
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["Description"] = df["Description"].astype(str).str.strip()
df = df[df["Description"].str.len() > 0]
df = df[df["Country"] == "United Kingdom"]  # keep the dominant, coherent market
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# ---------- 1) Market basket transactions ----------
top_products = df["Description"].value_counts().head(120).index
basket_df = df[df["Description"].isin(top_products)].copy()

# keep a manageable number of invoices, but require baskets with >=2 items
sizes = basket_df.groupby("InvoiceNo")["Description"].transform("nunique")
basket_df = basket_df[sizes >= 2]
keep_invoices = basket_df["InvoiceNo"].drop_duplicates().sample(
    n=min(4000, basket_df["InvoiceNo"].nunique()), random_state=42
)
basket_df = basket_df[basket_df["InvoiceNo"].isin(keep_invoices)]

basket_out = basket_df[["InvoiceNo", "InvoiceDate", "StockCode", "Description",
                          "Quantity", "UnitPrice", "CustomerID", "Country"]].copy()
basket_out = basket_out.sort_values(["InvoiceNo"]).reset_index(drop=True)
BASKET_OUT.parent.mkdir(parents=True, exist_ok=True)
basket_out.to_csv(BASKET_OUT, index=False)
print("basket_transactions:", basket_out.shape, "invoices:", basket_out["InvoiceNo"].nunique(),
      "products:", basket_out["Description"].nunique())

# ---------- 2) RFM customer table ----------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    recency_days=("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
    frequency=("InvoiceNo", "nunique"),
    monetary=("TotalPrice", "sum"),
    avg_basket_value=("TotalPrice", lambda s: s.sum() / max(1, df.loc[s.index, "InvoiceNo"].nunique())),
    distinct_products=("Description", "nunique"),
    first_purchase=("InvoiceDate", "min"),
    last_purchase=("InvoiceDate", "max"),
).reset_index()
rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days.clip(lower=1)
rfm["monetary"] = rfm["monetary"].round(2)
rfm["avg_basket_value"] = rfm["avg_basket_value"].round(2)
rfm = rfm[rfm["monetary"] > 0]
RFM_OUT.parent.mkdir(parents=True, exist_ok=True)
rfm.to_csv(RFM_OUT, index=False)
print("retail_rfm:", rfm.shape)
