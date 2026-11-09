"""
Speed Limit - MVP GUI.

Usage:  streamlit run app.py
Air quality reductions come from the local LLM if LM Studio is running, otherwise from the fallback rule.
"""
from pathlib import Path

import joblib
import streamlit as st

from air_quality import aqi_category, ask_llm, fetch_forecast, full_decision
from speedlimit_core import is_dark, is_ice_risk


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).with_name("speedlimit_model.joblib"))


@st.cache_data(ttl=600)
def live_air():
    return fetch_forecast()


@st.cache_data(show_spinner="Asking the local LLM...")
def cached_llm(aqi, pm2_5, pm10, no2, o3):
    """Remembers the answer, so changing the road conditions does not ask the LLM again."""
    return ask_llm(dict(aqi=aqi, pm2_5=pm2_5, pm10=pm10, no2=no2, o3=o3))


st.set_page_config(page_title="Speed Limit", page_icon="🚗")
st.title("Speed Limit")

st.subheader("Road conditions")
col1, col2, col3 = st.columns(3)
light = col1.number_input("Light (dark below 500)", min_value=0, value=50000, step=100)
water = col2.number_input("Water on road (µm)", min_value=0, value=0, step=100)
temp = col3.number_input("Temperature (°C)", min_value=-30, max_value=70, value=10)

st.subheader("Air quality")
source = st.radio("Source", ["Live forecast for Krems (Open-Meteo)", "Manual values"], horizontal=True)
air = None
if source.startswith("Live"):
    try:
        air = live_air()
    except Exception:
        st.warning("The forecast is unavailable. Choose \"Manual values\".")
else:
    cols = st.columns(5)
    air = {
        "aqi": cols[0].number_input("European AQI", min_value=0, max_value=500, value=92),
        "pm2_5": cols[1].number_input("PM2.5 (µg/m³)", min_value=0, max_value=1000, value=60),
        "pm10": cols[2].number_input("PM10 (µg/m³)", min_value=0, max_value=1000, value=120),
        "no2": cols[3].number_input("NO₂ (µg/m³)", min_value=0, max_value=1000, value=200),
        "o3": cols[4].number_input("Ozone (µg/m³)", min_value=0, max_value=1000, value=150),
    }

if air is None:
    st.stop()

result = full_decision(load_model(), light, water, temp, air, llm=lambda a: cached_llm(**a))

st.divider()
st.metric("Speed limit", f"{result['limit']} km/h")

reasons = [name for name, active in [("dark", is_dark(light)), ("ice risk", is_ice_risk(water, temp))] if active]
if reasons:
    note = "" if result["safe_limit_found"] else " No limit met the target of 1 near-accident per hour, so the 40 km/h fallback applies."
    st.write(f"**Weather:** {' and '.join(reasons)}, so the neural network chose {result['weather_limit']} km/h.{note}")
    st.caption("Predicted near-accidents per hour for each limit")
    st.table({"Limit (km/h)": list(result["predictions"]), "Predicted": list(result["predictions"].values())})
else:
    st.write("**Weather:** no reason to reduce, 80 km/h.")

decided_by = "local LLM" if result["source"] == "llm" else "fallback rule (LLM unavailable or invalid answer)"
st.write(f"**Air quality:** AQI {air['aqi']} ({aqi_category(air['aqi'])}), PM2.5 {air['pm2_5']}, PM10 {air['pm10']}, "
         f"NO₂ {air['no2']}, ozone {air['o3']} µg/m³. Reduction {result['reduction']} km/h, decided by the {decided_by}.")
