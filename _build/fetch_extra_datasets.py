"""One-off script: download the four datasets used as-is (no cleaning
transformation needed) by projects 05, 07, and 08, directly into each
project's data/ folder."""
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

FILES = [
    ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
     ROOT / "05_time_series_forecasting" / "data" / "airline_passengers.csv"),
    ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
     ROOT / "05_time_series_forecasting" / "data" / "daily_min_temperatures.csv"),
    ("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
     ROOT / "07_data_science_visual_foundations" / "data" / "sms_spam.tsv"),
    ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
     ROOT / "08_nano_transformer_llm" / "data" / "tinyshakespeare.txt"),
]

for url, dest in FILES:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

print("done.")
