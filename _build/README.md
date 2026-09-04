# Data build scripts

These are the one-off scripts used to turn public, real-world source data into
the compact CSVs shipped in each project's `data/` folder. They are **not**
needed to run the apps (the built CSVs are already committed) — they're here
for transparency and reproducibility, documenting exactly where every dataset
came from and how it was cleaned.

| Script | Produces | Source |
|---|---|---|
| `build_zone_centroids.py` | `taxi_zone_centroids.csv` (intermediate) | NYC TLC official taxi zone shapefile + lookup table (`d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip`, `.../taxi_zone_lookup.csv`) |
| `build_nyc_taxi_dataset.py` | `01_nyc_taxi_trip_duration/data/nyc_taxi_trips.csv` | NYC TLC official Yellow Taxi trip records, Jan 2023 (`d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet`) |
| `build_retail_datasets.py` | `02_customer_segmentation_clustering/data/retail_rfm.csv`, `03_market_basket_mining/data/basket_transactions.csv` | UCI "Online Retail" dataset (`archive.ics.uci.edu/ml/machine-learning-databases/00352/Online Retail.xlsx`) |
| `build_fraud_dataset.py` | `04_fraud_anomaly_detection/data/card_transactions.csv` | Kaggle "Credit Card Fraud Detection" (ULB Machine Learning Group) |
| `fetch_extra_datasets.py` | `05_time_series_forecasting/data/*.csv`, `07_data_science_visual_foundations/data/sms_spam.tsv`, `08_nano_transformer_llm/data/tinyshakespeare.txt` | Classic Box-Jenkins Airline Passengers & Melbourne daily temperatures (Jason Brownlee's public dataset mirror), UCI SMS Spam Collection, Andrej Karpathy's Tiny Shakespeare corpus |

`02_customer_segmentation_clustering/data/Mall_Customers.csv` is used as-is —
the classic, small (200-row) Kaggle "Mall Customers" demographic dataset,
no transformation needed. The four files `fetch_extra_datasets.py` downloads
are likewise used as-is (no cleaning/feature-engineering step) — they're
copied directly into each project's `data/` folder since they're already
small, real, and ready to use.

### Why re-derive instead of shipping raw source files

The raw sources (a 102MB fraud CSV, a 23MB retail spreadsheet, a 47MB monthly
taxi parquet, a taxi-zone shapefile) are excluded from this repo to keep it
lightweight. Each script re-downloads its source directly from the official
public URL, cleans it, and writes only the small, feature-engineered CSV each
Streamlit app actually reads — every dataset here is 100% traceable back to a
real, public, well-known source, never synthetic.

To regenerate everything from scratch:

```bash
cd _build
python build_zone_centroids.py      # writes taxi_zone_centroids.csv here
python build_nyc_taxi_dataset.py    # needs taxi_zone_centroids.csv above
python build_retail_datasets.py
python build_fraud_dataset.py
python fetch_extra_datasets.py
```

Each script downloads its own raw input on first run (see the source URLs
above / inside each script's docstring).
