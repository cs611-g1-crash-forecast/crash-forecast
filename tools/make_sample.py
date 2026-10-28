"""Build the data sample that ships with the submission: the five sources cut to the window in
config/pipeline.sample.yaml, written in the same layout and formats as the full data, so the
pipeline runs on it unchanged. Run from the repo folder, inside the image:
docker compose run --rm --no-deps jupyter python tools/make_sample.py <output folder>
"""
import csv
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yaml

with open("config/pipeline.yaml") as f:
    full = yaml.safe_load(f)
with open("config/pipeline.sample.yaml") as f:
    sample = yaml.safe_load(f)

raw_dir, out_dir = full["paths"]["raw_data"], sys.argv[1]
start = datetime.fromisoformat(sample["window"]["start"])
end = datetime.fromisoformat(sample["window"]["end"])
# a day's margin on each side: arrival times are compared as text here, while the pipeline converts
# time zones (weather is in UTC); rows outside the window are never read
first, last = f"{start - timedelta(days=1):%Y-%m-%d}", f"{end + timedelta(days=1):%Y-%m-%d}"
months = [f"{m:%Y-%m}" for m in pd.date_range(start, end, freq="MS")]

# only the columns the pipeline reads, for the two sources that are too large to ship whole
KEEP = {
    "tracker": ["time", "segment_id", "speed", "start_latitude", "start_longitude", "end_latitude", "end_longitude"],
    "taxi": ["trip_end_timestamp", "pickup_community_area"],
}


def write(df, name):
    path = os.path.join(out_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, compression="gzip")
    print(f"{name}: {len(df):,} rows, {os.path.getsize(path) / 1e6:.1f} MB")


for source in ["crashes", "weather", "tracker", "taxi", "licences"]:
    cfg = full["sources"][source]
    # the arrival column as the raw file names it (bronze renames weather's DATE to DATE_UTC)
    arrived = {new: old for old, new in cfg.get("rename", {}).items()}.get(cfg["arrived_at"], cfg["arrived_at"])
    names = [cfg["file"].format(month=m) for m in months] if "{month}" in cfg["file"] else [cfg["file"]]
    for name in names:
        df = pd.read_csv(os.path.join(raw_dir, name), dtype=str, keep_default_na=False, usecols=KEEP.get(source))
        if source == "licences":
            # the first run loads every licence issued before the window: they can still be active in it
            df = df[df[arrived] < last]
        else:
            df = df[(df[arrived] >= first) & (df[arrived] < last)]
        if source == "tracker":
            # the earliest reading of each segment in each hour (about one in six): enough for the
            # hourly congestion features, and sorted by segment so the file compresses well
            df = (df.assign(hour=df["time"].str[:13]).sort_values(["segment_id", "time"])
                    .drop_duplicates(["segment_id", "hour"]).drop(columns="hour"))
        # single-file sources take the sample's own file name, which says the dates it covers
        out_name = sample["sources"][source]["file"] if "{month}" not in cfg["file"] else name
        write(df, out_name)
