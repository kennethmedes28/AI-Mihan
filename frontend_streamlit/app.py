"""Flood-Gentic — NCR 7-day flood and weather dashboard.

Run from the repository root:

    streamlit run frontend_streamlit/app.py

Data comes from ``scripts/data_loader.py`` (Supabase tables
``ncr_flood_features_daily``, ``ncr_weather_daily``, ``ncr_locations``).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / "backend" / ".env")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import data_loader as dl  # noqa: E402

st.set_page_config(page_title="Flood-Gentic — NCR 7-Day Dashboard", layout="wide")

st.title("Flood-Gentic — NCR 7-Day Dashboard")
st.caption("Recent rain, river discharge, and flood features from Supabase. "
           "Averages are across sample points in each city; discharge and wet streak use the highest point.")

# --- Sidebar controls -------------------------------------------------------
locations = dl.load_locations()
city_names = ["All cities"] + sorted(locations["city"].tolist()) if not locations.empty else ["All cities"]

with st.sidebar:
    st.header("Filters")
    selected_label = st.selectbox("City", city_names, index=0)
    days = st.slider("Days to show", min_value=3, max_value=14, value=7, step=1)
    st.caption("Flood features and weather share the same day range.")

selected_city = None if selected_label == "All cities" else selected_label.lower()

# --- Load data --------------------------------------------------------------
try:
    flood = dl.load_recent_flood_features(days=days, city=selected_city)
    weather = dl.load_recent_weather(days=days, city=selected_city)
except Exception as exc:  # Supabase unreachable, missing tables, bad env, ...
    st.error(f"Could not load data from Supabase: {exc}")
    st.info("Check that backend/.env has the Supabase connection settings, then reload.")
    st.stop()

if flood.empty and weather.empty:
    st.warning("No rows returned for this selection. The dataset tables may be empty.")
    st.stop()

# --- Summary metrics --------------------------------------------------------
latest_date = flood["date"].max() if not flood.empty else weather["date"].max()
cities_covered = flood["city"].nunique() if not flood.empty else weather["city"].nunique()

st.subheader(f"Summary — up to {latest_date}")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Cities", int(cities_covered))
with col2:
    st.metric("Days", int(days))
with col3:
    if not flood.empty:
        latest = flood[flood["date"] == latest_date]
        st.metric("Wettest city (latest day)", f"{latest.loc[latest['rain_mm'].idxmax(), 'city']} "
                  f"({latest['rain_mm'].max():.1f} mm)")
    else:
        st.metric("Wettest city (latest day)", "—")
with col4:
    if not flood.empty:
        latest = flood[flood["date"] == latest_date]
        st.metric("Highest discharge (latest day)", f"{latest['river_discharge_m3s'].max():.2f} m³/s")
    else:
        st.metric("Highest discharge (latest day)", "—")

# --- Charts -----------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Daily rainfall (mm)")
    if not flood.empty:
        if selected_city:
            chart = flood.set_index("date")[["rain_mm"]]
        else:
            chart = flood.pivot(index="date", columns="city", values="rain_mm")
        st.line_chart(chart)
    else:
        st.write("No flood-feature data.")

with right:
    st.subheader("River discharge (m³/s)")
    if not flood.empty:
        if selected_city:
            chart = flood.set_index("date")[["river_discharge_m3s"]]
        else:
            chart = flood.pivot(index="date", columns="city", values="river_discharge_m3s")
        st.line_chart(chart)
    else:
        st.write("No flood-feature data.")

with left:
    st.subheader("Temperature (°C)")
    if not weather.empty:
        if selected_city:
            chart = weather.set_index("date")[["temp_max_c", "temp_min_c"]]
        else:
            daily = weather.groupby("date")[["temp_max_c", "temp_min_c"]].mean()
            chart = daily
        st.line_chart(chart)
    else:
        st.write("No weather data.")

with right:
    st.subheader("Wind — max and gusts (km/h)")
    if not weather.empty:
        if selected_city:
            chart = weather.set_index("date")[["wind_max_kmh", "wind_gust_kmh"]]
        else:
            chart = weather.groupby("date")[["wind_max_kmh", "wind_gust_kmh"]].mean()
        st.line_chart(chart)
    else:
        st.write("No weather data.")

# --- Map --------------------------------------------------------------------
st.subheader("NCR sample points")
if not locations.empty:
    map_data = locations.rename(columns={"latitude": "lat", "longitude": "lon"})
    st.map(map_data[["lat", "lon"]])
    st.caption("One point per city (17 cities). Nearby cities can share weather on the Open-Meteo grid.")
else:
    st.write("No location data.")

# --- Tables -----------------------------------------------------------------
st.subheader("Flood features")
if not flood.empty:
    st.dataframe(flood, use_container_width=True, hide_index=True)
else:
    st.write("No flood-feature rows.")

st.subheader("Daily weather")
if not weather.empty:
    st.dataframe(weather, use_container_width=True, hide_index=True)
else:
    st.write("No weather rows.")

st.caption("Source: Supabase ncr_flood_features_daily / ncr_weather_daily / ncr_locations. "
           "Rain is averaged across points in a city; discharge and wet streak use the highest point. "
           "Cached for 30 minutes.")
