"""One-off script: build a repo-friendly stratified sample of the
Kaggle 'Credit Card Fraud Detection' dataset (mlg-ulb/ULB Machine
Learning Group), keeping every fraud and a random slice of legitimate
transactions, preserving the original class imbalance ratio at a
smaller, more tractable scale (~3.4% fraud rate)."""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Mirror of the Kaggle mlg-ulb/creditcardfraud dataset (Kaggle itself requires
# auth to download programmatically; this is a widely-used public GitHub mirror
# of the same file).
SOURCE_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
csv_path = HERE / "creditcard.csv"
OUT_PATH = HERE.parent / "04_fraud_anomaly_detection" / "data" / "card_transactions.csv"

if not csv_path.exists():
    print("Downloading creditcard.csv (~100MB) ...")
    urllib.request.urlretrieve(SOURCE_URL, csv_path)

RNG = 42
df = pd.read_csv(csv_path)

fraud = df[df["Class"] == 1].copy()
legit = df[df["Class"] == 0].sample(n=14000, random_state=RNG).copy()

out = pd.concat([fraud, legit]).sample(frac=1.0, random_state=RNG).reset_index(drop=True)
# round PCA components for a smaller file
for c in [c for c in out.columns if c.startswith("V")]:
    out[c] = out[c].round(4)
out["Amount"] = out["Amount"].round(2)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(out.shape, out["Class"].mean())
