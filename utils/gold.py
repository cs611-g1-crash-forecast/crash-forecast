import os
from datetime import datetime, timedelta

import pandas as pd
import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pyspark.sql import Window

from utils.bronze import run_id_for

AREAS = range(1, 78)  # the 77 community areas


def hour_grid(first, last):
    """Every Chicago clock hour from first to last, inclusive, as naive timestamps.
    The spring-forward hour doesn't exist, so it has no row; the fall-back hour happens twice
    but is one row."""
    # an end point on the fall-back hour is ambiguous, and one in the spring gap doesn't exist:
    # take the range as wide as those clock times allow
    first = pd.Timestamp(first).tz_localize("America/Chicago", ambiguous=True, nonexistent="shift_forward")
    last = pd.Timestamp(last).tz_localize("America/Chicago", ambiguous=False, nonexistent="shift_backward")
    hours = pd.date_range(first, last, freq="h").tz_localize(None)
    return hours.drop_duplicates()


def grid_df(hours, spark):
    """The full grid: one row per community area x hour."""
    pdf = pd.DataFrame([(area, hour) for hour in hours for area in AREAS],
                       columns=["community_area", "hour_start"])
    return spark.createDataFrame(pdf).select(
        F.col("community_area").cast("int"), F.col("hour_start").cast("timestamp_ntz"))


def hour_of(column):
    """The clock hour a timestamp falls in (date_trunc returns a zoned timestamp: cast it back)."""
    return F.date_trunc("hour", column).cast("timestamp_ntz")


def is_bulk(start, config):
    return start < datetime.fromisoformat(config["deployment"]["go_live"])


def read_silver(source, first_date, last_date, config, spark):
    """Silver rows that arrived from first_date to last_date (dates, inclusive); reads only those partitions."""
    path = os.path.join(config["paths"]["datamart"], "silver", source)
    return spark.read.parquet(path).filter(F.col("date").between(first_date, last_date))


def write_gold(df, table, run_id, config, spark):
    # partitioned by the hour's date, then run; a rerun replaces only its own partitions
    gold_path = os.path.join(config["paths"]["datamart"], "gold", table)
    (df.withColumn("date", F.to_date("hour_start"))
       .withColumn("run_id", F.lit(run_id))
       .write.mode("overwrite").partitionBy("date", "run_id").parquet(gold_path))
    written = spark.read.parquet(gold_path).filter(F.col("run_id") == run_id).count()
    return gold_path, written


def label_hours(start, end, config):
    """The hours a gold_labels run labels, [first, end_hour), and its arrival cutoff.
    Bulk (a month): this month, and the month before again, now with a month more of late reports.
    Live (a day): the day before, so every hour has had at least label_lag_days for its reports."""
    if is_bulk(start, config):
        window_start = datetime.fromisoformat(config["window"]["start"])
        first = max(start - relativedelta(months=1), window_start)
        return first, end, end
    lag = timedelta(days=config["monitoring"]["label_lag_days"])
    return start - lag, end - lag, end


def process_gold_labels(start, end, config, spark):
    run_id = run_id_for(start, end)
    first, end_hour, cutoff = label_hours(start, end, config)

    # crashes that occurred in the hours being labelled and were reported before the cutoff
    crashes = (read_silver("crashes", first.date(), cutoff.date(), config, spark)
               .filter(F.col("_arrived_at") < cutoff)
               .withColumn("hour_start", hour_of("crash_date"))
               .filter((F.col("hour_start") >= first) & (F.col("hour_start") < end_hour)))
    counts = crashes.groupBy("community_area", "hour_start").agg(
        F.count("*").alias("crash_count"),
        F.sum(F.when(F.col("crash_type").startswith("INJURY"), 1).otherwise(0)).alias("crash_count_injury_tow"))

    # the full grid, with zeros kept explicitly
    grid = grid_df(hour_grid(first, end_hour - timedelta(hours=1)), spark)
    labels = (grid.join(counts, ["community_area", "hour_start"], "left")
              .fillna(0, ["crash_count", "crash_count_injury_tow"])
              .select("community_area", "hour_start",
                      F.col("crash_count").cast("int"), F.col("crash_count_injury_tow").cast("int"),
                      F.lit(cutoff.date()).alias("label_snapshot_date")))

    # save gold table to datamart - IRL write to the data lake (S3)
    gold_path, written = write_gold(labels, "label_store", run_id, config, spark)
    total = labels.agg(F.sum("crash_count")).first()[0]
    print(f"gold labels {run_id}: hours {first:%Y-%m-%d %H:%M} to {end_hour:%Y-%m-%d %H:%M}, crashes reported "
          f"before {cutoff:%Y-%m-%d}: {total:,}; {written:,} rows written to {gold_path}")
    return written


def target_hours(start, end, config):
    """The target hours T a gold_features run writes: (start, end].
    A run has seen every arrival before end, so it can build features up to T = end:
    the hourly run for [T-1, T) writes T's 77 rows; the first run also writes the window's first hour."""
    window_start = datetime.fromisoformat(config["window"]["start"])
    first = start if start <= window_start else start + timedelta(hours=1)
    return hour_grid(first, end)


def cal_features(targets, config, spark):
    """cal_: known in advance for any hour."""
    holidays = set(USFederalHolidayCalendar().holidays(targets.min(), targets.max()).date)
    covid = config["features"]["covid_regime"]
    pdf = pd.DataFrame({"hour_start": targets})
    pdf["cal_hour"] = pdf["hour_start"].dt.hour
    pdf["cal_weekday"] = pdf["hour_start"].dt.dayofweek + 1  # 1 = Monday
    pdf["cal_month"] = pdf["hour_start"].dt.month
    pdf["cal_is_holiday"] = pdf["hour_start"].dt.date.isin(holidays).astype(int)
    month = pdf["hour_start"].dt.strftime("%Y-%m")
    pdf["cal_covid_regime"] = month.between(covid["start"], covid["end"]).astype(int)
    df = spark.createDataFrame(pdf)
    return df.select(F.col("hour_start").cast("timestamp_ntz"),
                     *[F.col(c).cast("double") for c in pdf.columns if c != "hour_start"])


def crash_features(targets, config, spark):
    """crash_: what was known about crashes before T (by date_police_notified = _arrived_at)."""
    first, last = targets.min().to_pydatetime(), targets.max().to_pydatetime()
    crashes = read_silver("crashes", (first - timedelta(days=29)).date(), last.date(), config, spark)

    # 1. crashes notified in the last 1 h / 3 h / 24 h before T.
    # A crash notified at a counts for the 24 targets after it: T = the next hour start after a, + 0..23 h
    notified = (crashes
                .withColumn("i", F.explode(F.sequence(F.lit(0), F.lit(23))))
                .withColumn("T", F.timestamp_add("HOUR", F.col("i") + 1, hour_of("_arrived_at")))
                .withColumn("seconds_before", F.col("T").cast("timestamp").cast("long")
                            - F.col("_arrived_at").cast("timestamp").cast("long")))
    windows = {"1h": 3600, "3h": 3 * 3600, "24h": 24 * 3600}
    sums = lambda scope: [F.sum((F.col("seconds_before") <= s).cast("int")).cast("double")
                          .alias(f"crash_{scope}_notified_{w}") for w, s in windows.items()]
    area_notified = notified.groupBy("community_area", "T").agg(*sums("area"))
    city_notified = notified.groupBy("T").agg(*sums("city"))

    # 2. the mean count for the same hour of the week over the previous 4 weeks (B0's forecast).
    # A crash that occurred in hour o counts for T = o + 7, 14, 21, 28 days, but only if it was notified before T
    weekly = (crashes
              .withColumn("k", F.explode(F.sequence(F.lit(1), F.lit(4))))
              .withColumn("T", F.timestamp_add("DAY", F.col("k") * 7, hour_of("crash_date")))
              .filter(F.col("_arrived_at") < F.col("T")))
    area_weekly = weekly.groupBy("community_area", "T").agg((F.count("*") / 4).alias("crash_area_how_mean_4w"))
    city_weekly = weekly.groupBy("T").agg((F.count("*") / 4).alias("crash_city_how_mean_4w"))

    grid = grid_df(targets, spark).withColumnRenamed("hour_start", "T")
    df = (grid.join(area_notified, ["community_area", "T"], "left")
              .join(city_notified, ["T"], "left")
              .join(area_weekly, ["community_area", "T"], "left")
              .join(city_weekly, ["T"], "left"))
    feature_columns = [c for c in df.columns if c.startswith("crash_")]
    df = df.fillna(0.0, feature_columns)

    # the 4-week means need 4 weeks of history: missing before then, not zero
    window_start = datetime.fromisoformat(config["window"]["start"])
    for c in ["crash_area_how_mean_4w", "crash_city_how_mean_4w"]:
        df = df.withColumn(c, F.when(F.col("T") >= window_start + timedelta(days=28), F.col(c)))
    return df.withColumnRenamed("T", "hour_start")


def wx_features(targets, config, spark):
    """wx_: the last FM-15 reading observed before T (citywide, at O'Hare)."""
    first, last = targets.min().to_pydatetime(), targets.max().to_pydatetime()
    weather = read_silver("weather", (first - timedelta(days=2)).date(), last.date(), config, spark)
    hours = spark.createDataFrame(pd.DataFrame({"T": targets})).select(F.col("T").cast("timestamp_ntz"))

    # each target with the readings of the 48 h before it; keep the latest
    candidates = hours.join(weather, (weather["observed_at"] < hours["T"])
                            & (weather["observed_at"] >= F.timestamp_add("HOUR", F.lit(-48), hours["T"])))
    latest = Window.partitionBy("T").orderBy(F.col("observed_at").desc())
    last_reading = (candidates.withColumn("n", F.row_number().over(latest)).filter("n = 1")
                    .select(F.col("T").alias("hour_start"),
                            F.col("temp_c").alias("wx_temp_c"),
                            F.col("dew_point_c").alias("wx_dew_point_c"),
                            F.col("visibility_m").alias("wx_visibility_m"),
                            F.col("wind_ms").alias("wx_wind_ms"),
                            F.col("precip_1h_mm").alias("wx_precip_1h_mm"),
                            ((F.col("T").cast("timestamp").cast("long")
                              - F.col("observed_at").cast("timestamp").cast("long")) / 3600)
                            .alias("wx_age_hours")))
    # no reading in the 48 h before T: the wx_ features stay null
    return hours.withColumnRenamed("T", "hour_start").join(last_reading, ["hour_start"], "left")


def process_gold_features(start, end, config, spark):
    run_id = run_id_for(start, end)
    targets = target_hours(start, end, config)

    from utils.licences import lic_features
    from utils.taxi import taxi_features
    from utils.tracker import trk_features
    
    families = {"cal": cal_features, "crash": crash_features, "wx": wx_features,
                "trk": trk_features, "taxi": taxi_features, "lic": lic_features} 
    df = grid_df(targets, spark).withColumn("feature_cutoff", F.col("hour_start"))
    for family in config["features"]["families"]:
        family_df = families[family](targets, config, spark)
        keys = ["community_area", "hour_start"] if "community_area" in family_df.columns else ["hour_start"]
        df = df.join(family_df, keys, "left")
    # a fixed column order: keys, cutoff, then the features by name
    df = df.select("community_area", "hour_start", "feature_cutoff",
                   *sorted(c for c in df.columns if c not in ("community_area", "hour_start", "feature_cutoff")))

    # save gold table to datamart - IRL write to the data lake (S3)
    gold_path, written = write_gold(df, "feature_store", run_id, config, spark)
    print(f"gold features {run_id}: {len(targets):,} target hours from {targets.min():%Y-%m-%d %H:%M} "
          f"to {targets.max():%Y-%m-%d %H:%M}, {len(df.columns) - 3} features; "
          f"{written:,} rows written to {gold_path}")
    return written
