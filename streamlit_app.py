"""Run with: python -m streamlit run streamlit_app.py"""

import json
from datetime import date

import pandas as pd
import streamlit as st

from planner_core import load_places, plan_day


st.set_page_config(page_title="Rourkela Travel Planner", page_icon="🧭", layout="wide")
st.title("🧭 Rourkela Travel Planner")
st.caption("Explore local attractions and make a simple day route. Travel times assume 20 km/h and one hour at each stop.")

places = load_places()
titles = [place["title"] for place in places]
start_locations = {"Rourkela city centre": (22.2396, 84.8633)}
start_locations.update({place["title"]: (float(place["lat"]), float(place["lng"])) for place in places})

with st.sidebar:
    st.header("Plan your day")
    visit_date = st.date_input("Date", value=date.today())
    start_name = st.selectbox("Starting from", list(start_locations))
    radius = st.slider("Search radius (km)", 1, 60, 25)
    start_hour = st.slider("Start hour", 0, 22, 8)
    end_hour = st.slider("End hour", 1, 23, 20)
    max_stops = st.slider("Maximum stops", 1, 9, 5)
    selected = st.multiselect("Specific places (optional)", titles)
    generate = st.button("Create itinerary", type="primary", use_container_width=True)

if generate:
    try:
        center = start_locations[start_name]
        nearby, itinerary = plan_day(places, center, visit_date, start_hour, end_hour, radius, max_stops, selected)
        st.session_state["plan"] = (nearby, itinerary, visit_date.isoformat(), center, start_name)
    except ValueError as exc:
        st.error(str(exc))

if "plan" in st.session_state:
    nearby, itinerary, planned_date, center, planned_start = st.session_state["plan"]
    st.subheader(f"Itinerary for {planned_date}")
    st.caption(f"Starting from {planned_start}")
    if not itinerary:
        st.warning("No stops fit this route and time window. Increase the radius or available hours.")
    else:
        for stop in itinerary:
            with st.container(border=True):
                st.markdown(f"**{stop['order']}. {stop['place']}** · {stop['time']}")
                st.caption(f"{stop['distance_km']} km from previous point · about {stop['travel_minutes']} min travel")
                st.write(stop["description"])
        map_rows = [{"lat": center[0], "lon": center[1], "name": "Start"}]
        map_rows += [{"lat": p["lat"], "lon": p["lng"], "name": p["place"]} for p in itinerary]
        st.map(pd.DataFrame(map_rows), latitude="lat", longitude="lon")
        public_schedule = [{key: value for key, value in stop.items() if key not in ("lat", "lng")} for stop in itinerary]
        st.download_button("Download itinerary JSON", json.dumps({"date": planned_date, "starting_from": planned_start, "schedule": public_schedule}, indent=2), file_name="rourkela_itinerary.json", mime="application/json")
    with st.expander(f"Nearby places ({len(nearby)})"):
        st.dataframe(pd.DataFrame([{"Place": p["title"], "Distance (km)": p["distance_from_start_km"]} for p in nearby]), hide_index=True)
else:
    st.info("Choose your route settings in the sidebar, then click Create itinerary.")
    st.map(pd.DataFrame([{"lat": float(p["lat"]), "lon": float(p["lng"])} for p in places]), latitude="lat", longitude="lon")
