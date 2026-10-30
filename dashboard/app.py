"""Planner dashboard: where to stage ambulances for a shift, from the production model's hourly forecasts.

A service of its own, not an Airflow task: it rereads the gold tables every dashboard.refresh_seconds
and writes nothing. Run from the repo root: streamlit run dashboard/app.py"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st
import yaml

# streamlit puts dashboard/ on the path; utils/, config/ and datamart/ are at the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dashboard.data import (areas, forecast_dates, labels, production_forecasts, rank_areas,  # noqa: E402
                            shift_result, shift_start)

with open("config/pipeline.yaml") as f:
    CONFIG = yaml.safe_load(f)

REFRESH = CONFIG["dashboard"]["refresh_seconds"]
K, SHIFT_HOURS = CONFIG["evaluation"]["top_k"], CONFIG["evaluation"]["shift_hours"]
# one-hue sequential ramp, light (few crashes) to dark (many)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

st.set_page_config(page_title="Crash forecast: planner", page_icon="🚑", layout="wide")


@st.cache_data(ttl=REFRESH, show_spinner=False)
def load_dates():
    return forecast_dates(CONFIG)


@st.cache_data(ttl=REFRESH, show_spinner=False)
def load_day(day):
    return production_forecasts(day, CONFIG), labels(day, CONFIG)


@st.cache_data(show_spinner=False)
def load_areas():
    return areas(CONFIG)


def read_day(day):
    """The day's forecasts and labels; if a run is writing them right now, the last good read."""
    good = st.session_state.setdefault("last_good", {})
    try:
        good[day] = (*load_day(day), datetime.now())
    except Exception as error:  # e.g. a parquet file read while it was being written
        if day not in good:
            st.error(f"Couldn't read the forecasts for {day}: {error}")
            st.stop()
        st.warning(f"Couldn't read the tables just now (a run may be writing them); "
                   f"showing the data as of {good[day][2]:%H:%M:%S}.")
    return good[day][:2]


def colour(share):
    """The ramp's colour for a forecast as a share of the map's highest, as [r, g, b]."""
    hex_colour = RAMP[min(int(share * (len(RAMP) - 1) + 0.5), len(RAMP) - 1)]
    return [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]


def area_map(ranked, top):
    highest = max(ranked["forecast"].max(), 1e-9)
    by_area = ranked.set_index("community_area")
    features = []
    for number, (name, geometry) in load_areas().items():
        row = by_area.loc[number]
        features.append({"type": "Feature", "geometry": geometry, "properties": {
            "name": name, "number": number, "rank": int(row["rank"]), "forecast": f"{row['forecast']:.2f}",
            "fill": colour(row["forecast"] / highest), "top": number in top}})
    theme = "dark" if st.context.theme.type == "dark" else "light"
    outline = [255, 255, 255] if theme == "dark" else [33, 33, 33]
    fill = pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": features}, pickable=True,
                     filled=True, stroked=True, get_fill_color="properties.fill", opacity=0.85,
                     get_line_color=[255, 255, 255], line_width_min_pixels=0.5)
    # the staged areas outlined in a layer of their own, so a neighbour's fill never covers the outline
    staged = pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection",
                                        "features": [f for f in features if f["properties"]["top"]]},
                       filled=False, stroked=True, get_line_color=outline, line_width_min_pixels=3)
    return pdk.Deck(layers=[fill, staged], map_style=theme,
                    initial_view_state=pdk.ViewState(latitude=41.835, longitude=-87.69, zoom=9.6),
                    tooltip={"html": "<b>{name}</b> (area {number})<br/>rank {rank} · forecast {forecast}"})


def legend(highest):
    gradient = ", ".join(RAMP)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;font-size:0.8rem'>"
        f"<span>0</span><div style='flex:1;height:10px;border-radius:4px;"
        f"background:linear-gradient(to right, {gradient})'></div><span>{highest:.2f}</span></div>"
        f"<div style='font-size:0.8rem;opacity:0.7'>crashes forecast in the hour · "
        f"outlined: the top {K} areas to stage in</div>", unsafe_allow_html=True)


# --- which hour to show: the newest forecast (follows the replay), or one picked by hand

dates = load_dates()
if not dates:
    st.title("Planner")
    st.info("No forecasts yet: run hourly_forecast or replay_days, and this page fills in by itself.")
    st.stop()

newest_forecasts, _ = read_day(dates[-1])
newest = newest_forecasts["hour_start"].max()


@st.fragment(run_every=REFRESH)
def watch_for_new_forecasts():
    """Reruns the whole page when the pipeline has written a newer hour. Reads past the cache, so the
    page moves on within one refresh instead of waiting for the cached tables to expire as well."""
    try:
        latest = forecast_dates(CONFIG)[-1]
        moved = latest != dates[-1] or production_forecasts(latest, CONFIG)["hour_start"].max() != newest
    except Exception:  # a run writing the tables right now: look again next time
        return
    if moved:
        load_dates.clear()
        load_day.clear()
        st.rerun()


with st.sidebar:
    st.header("Shift")
    follow = st.toggle("Follow the newest forecast", value=True,
                       help="The page moves on by itself as the pipeline forecasts each new hour.")
    shift_starts = list(range(0, 24, SHIFT_HOURS))
    if follow:
        hour = newest
        first = shift_start(hour, SHIFT_HOURS)
        st.markdown(f"**{first:%a %d %b %Y}**, shift {first:%H:%M}–{first + timedelta(hours=SHIFT_HOURS):%H:%M}"
                    f"\n\nhour {hour:%H:%M} (the newest forecast)")
    else:
        day = st.date_input("Day", value=newest.date(), min_value=pd.Timestamp(dates[0]).date(),
                            max_value=pd.Timestamp(dates[-1]).date(), format="YYYY-MM-DD")
        start_hour = st.selectbox("Shift", shift_starts, index=shift_starts.index(shift_start(newest, SHIFT_HOURS).hour),
                                  format_func=lambda h: f"{h:02d}:00–{h + SHIFT_HOURS:02d}:00")
        first = pd.Timestamp(day) + timedelta(hours=start_hour)
    st.caption(f"Data refreshes every {REFRESH} s. Newest forecast: {newest:%d %b %Y %H:%M}.")

watch_for_new_forecasts()

day = f"{first:%Y-%m-%d}"
forecasts, day_labels = read_day(day) if day in dates else (pd.DataFrame(), pd.DataFrame())
last = first + timedelta(hours=SHIFT_HOURS)
in_shift = forecasts[(forecasts["hour_start"] >= first) & (forecasts["hour_start"] < last)] if len(forecasts) else forecasts

st.title(f"Where to stage: {first:%a %d %b %Y}, {first:%H:%M}–{last:%H:%M}")
if in_shift.empty:
    st.info("No production forecasts for this shift yet.")
    st.stop()

hours = sorted(in_shift["hour_start"].unique())
if not follow:
    hour = st.sidebar.select_slider("Hour in the shift", options=hours, value=hours[0],
                                    format_func=lambda h: f"{pd.Timestamp(h):%H:%M}",
                                    help="Each hour has its own forecast, so units can move during the shift.")
hour = pd.Timestamp(hour)

ranked = rank_areas(in_shift, hour, {n: name for n, (name, _) in load_areas().items()})
hour_rows = in_shift[in_shift["hour_start"] == hour]
top = set(ranked["community_area"].head(K))

a, b, c = st.columns(3)
a.metric(f"Crashes forecast citywide, {hour:%H:%M}–{hour + timedelta(hours=1):%H:%M}", f"{ranked['forecast'].sum():.1f}")
model_version = hour_rows["model_version"].iloc[0]
b.metric("Production model", model_version.split("_")[0], help=model_version)
c.metric("Forecast from data up to", f"{hour_rows['feature_cutoff'].max():%d %b %H:%M}")

left, right = st.columns([3, 2])
with left:
    st.pydeck_chart(area_map(ranked, top), height=620)
    legend(ranked["forecast"].max())
with right:
    st.subheader(f"Top {K} areas")
    st.dataframe(ranked.head(K).rename(columns={"rank": "Rank", "area": "Area", "community_area": "No.",
                                                "forecast": "Forecast"}),
                 hide_index=True, column_config={"Forecast": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("How the shift went")
    result = shift_result(in_shift, day_labels, first, SHIFT_HOURS, K) if len(day_labels) else None
    if result is None:
        st.caption("The shift's crash counts come in a day after it ends (monitoring.label_lag_days); "
                   "until then there is nothing to score.")
    else:
        x, y, z = st.columns(3)
        x.metric("Crashes in the shift", result["total"])
        y.metric(f"In the top {K}", result["caught"], help=f"Crashes in the {K} areas staged at the shift's start.")
        z.metric("P0", "–" if result["p0"] is None else f"{result['p0']:.2f}",
                 help=f"The share of the shift's crashes in the {K} areas forecast highest at its start.")
