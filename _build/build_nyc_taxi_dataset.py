"""One-off script: stream Jan-2023 NYC Yellow Taxi trip data from the
official TLC S3/CloudFront bucket (via fsspec, footer+row-group reads,
no full-file download), sample ~40k clean trips, join zone centroids,
engineer the CRISP-DM feature set, and write a compact CSV asset for the
Streamlit app.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import fsspec

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE.parent / "01_nyc_taxi_trip_duration" / "data" / "nyc_taxi_trips.csv"
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
RNG = np.random.default_rng(42)

COLS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "PULocationID", "DOLocationID", "payment_type",
    "fare_amount", "tip_amount", "total_amount", "RatecodeID",
]

print("Opening remote parquet (footer only)...")
fs = fsspec.filesystem("http")
with fs.open(URL, "rb") as f:
    pf = pq.ParquetFile(f)
    print("row groups:", pf.num_row_groups, "total rows:", pf.metadata.num_rows)
    # Read a handful of row groups spread across the file for temporal diversity
    n_rg = pf.num_row_groups
    picks = sorted(set(np.linspace(0, n_rg - 1, min(12, n_rg)).astype(int)))
    tables = [pf.read_row_group(i, columns=COLS) for i in picks]

import pyarrow as pa
df = pa.concat_tables(tables).to_pandas()
print("raw sample rows:", len(df))

# --- Clean (CRISP-DM: Data Preparation) ---
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
df["trip_duration_s"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()

df = df[
    (df["trip_duration_s"] > 60) & (df["trip_duration_s"] < 3 * 3600) &
    (df["trip_distance"] > 0.1) & (df["trip_distance"] < 50) &
    (df["passenger_count"].fillna(1) > 0) & (df["passenger_count"].fillna(1) <= 6) &
    (df["fare_amount"] > 0) & (df["fare_amount"] < 300) &
    (df["tpep_pickup_datetime"].dt.year == 2023) & (df["tpep_pickup_datetime"].dt.month == 1)
].copy()

zones = pd.read_csv(HERE / "taxi_zone_centroids.csv")
zmap = zones.set_index("LocationID")

df = df[df["PULocationID"].isin(zmap.index) & df["DOLocationID"].isin(zmap.index)]
df = df[(df["PULocationID"] != 264) & (df["PULocationID"] != 265) &
        (df["DOLocationID"] != 264) & (df["DOLocationID"] != 265)]  # drop Unknown/N/A zones

pu = zmap.loc[df["PULocationID"]]
do = zmap.loc[df["DOLocationID"]]
df["pickup_latitude"] = pu["centroid_lat"].values
df["pickup_longitude"] = pu["centroid_lon"].values
df["dropoff_latitude"] = do["centroid_lat"].values
df["dropoff_longitude"] = do["centroid_lon"].values
df["pickup_zone"] = pu["zone"].values
df["pickup_borough"] = pu["borough"].values
df["dropoff_zone"] = do["zone"].values
df["dropoff_borough"] = do["borough"].values

# jitter centroids slightly so points don't all stack exactly on one pixel (realistic
# within-zone spread, still bounded to the zone's neighborhood)
jitter = 0.004
df["pickup_latitude"] += RNG.normal(0, jitter, len(df))
df["pickup_longitude"] += RNG.normal(0, jitter, len(df))
df["dropoff_latitude"] += RNG.normal(0, jitter, len(df))
df["dropoff_longitude"] += RNG.normal(0, jitter, len(df))

# haversine distance (independent of trip_distance meter reading -> good feature)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

df["haversine_km"] = haversine(df["pickup_latitude"], df["pickup_longitude"],
                                 df["dropoff_latitude"], df["dropoff_longitude"])

df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.dayofweek
df["pickup_month_day"] = df["tpep_pickup_datetime"].dt.day
df["is_weekend"] = (df["pickup_weekday"] >= 5).astype(int)
df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df["is_night"] = ((df["pickup_hour"] >= 22) | (df["pickup_hour"] <= 5)).astype(int)

df = df.rename(columns={
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
})

keep = [
    "pickup_datetime", "dropoff_datetime", "passenger_count", "trip_distance",
    "haversine_km", "pu_location_id", "do_location_id", "pickup_zone",
    "pickup_borough", "dropoff_zone", "dropoff_borough",
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    "pickup_hour", "pickup_weekday", "pickup_month_day", "is_weekend",
    "is_rush_hour", "is_night", "payment_type", "RatecodeID",
    "fare_amount", "tip_amount", "total_amount", "trip_duration_s",
]
df = df[keep].dropna(subset=["passenger_count"])
df["passenger_count"] = df["passenger_count"].astype(int)

# Downsample to a manageable, class-diverse, repo-friendly size
n_target = 20000
if len(df) > n_target:
    df = df.sample(n=n_target, random_state=42).reset_index(drop=True)

df = df.sort_values("pickup_datetime").reset_index(drop=True)

for c in ["haversine_km", "pickup_latitude", "pickup_longitude",
          "dropoff_latitude", "dropoff_longitude"]:
    df[c] = df[c].round(5)
for c in ["trip_distance", "fare_amount", "tip_amount", "total_amount"]:
    df[c] = df[c].round(2)
df["trip_duration_s"] = df["trip_duration_s"].astype(int)

print("final rows:", len(df), "columns:", len(df.columns))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("saved.")
