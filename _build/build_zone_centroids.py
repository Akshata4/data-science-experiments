"""One-off script: compute lat/lon centroids for each NYC TLC taxi zone
from the official shapefile, and merge with the zone lookup table.
Downloads its raw inputs from the official TLC CloudFront bucket on
first run."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import shapefile  # pyshp
from pyproj import Transformer
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
ZONES_ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

zones_dir = HERE / "taxi_zones"
lookup_path = HERE / "taxi_zone_lookup.csv"

if not zones_dir.exists():
    print("Downloading taxi_zones.zip ...")
    data = urllib.request.urlopen(ZONES_ZIP_URL).read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(zones_dir)

if not lookup_path.exists():
    print("Downloading taxi_zone_lookup.csv ...")
    urllib.request.urlretrieve(LOOKUP_URL, lookup_path)

shp_path = next(zones_dir.rglob("*.shp"))
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

sf = shapefile.Reader(str(shp_path.with_suffix("")))
fields = [f[0] for f in sf.fields[1:]]

rows = []
for sr in sf.shapeRecords():
    rec = dict(zip(fields, sr.record))
    geom = shape(sr.shape.__geo_interface__)
    c = geom.centroid
    lon, lat = transformer.transform(c.x, c.y)
    rows.append({
        "LocationID": int(rec["LocationID"]),
        "zone": rec["zone"],
        "borough": rec["borough"],
        "centroid_lat": lat,
        "centroid_lon": lon,
    })

zones = pd.DataFrame(rows).sort_values("LocationID")
lookup = pd.read_csv(lookup_path)
merged = zones.merge(lookup[["LocationID", "service_zone"]], on="LocationID", how="left")
merged.to_csv(HERE / "taxi_zone_centroids.csv", index=False)
print(merged.shape)
print(merged.head())
