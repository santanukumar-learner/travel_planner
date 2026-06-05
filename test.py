# schedule_runner.py
import os
from datetime import datetime

from rag import RourkelalTourismSchedulePlanner
from place_recommender import LatLngSchedulePlanner  # your LatLngSchedulePlanner file


def run():
    # You can also hardcode keys here while testing if env vars are not set
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    WEATHER_API_KEY = os.getenv("WHEATHER_KEY")  # keep same env name you used in rag.py

    planner = RourkelalTourismSchedulePlanner(
        api_key=GEMINI_API_KEY,
        weather_api_key=WEATHER_API_KEY,
        attractions_file="rag.json",
        restaurants_file="restaurants_rourkela.json",
        model_file="rkl_places_xgb.pkl",
    )

    geo = LatLngSchedulePlanner(
        planner,
        default_radius_km=8.0,
        dwell_minutes=60,
        travel_speed_kmh=20.0,
    )

    # Center somewhere in Rourkela
    lat, lng = 22.2396, 84.8633

    # 👉 Add preferred places for testing (must match titles in rag.json)
    preferred = ["Hanuman Vatika", "Indira Gandhi Park"]

    day_plan = geo.plan_day(
        lat=lat,
        lng=lng,
        date=datetime.now(),
        radius_km=8.0,
        max_stops=6,
        start_hour=8,
        end_hour=20,
        preferred_places=preferred,   # ✅ test preference filter
        include_nearby=True,          # ✅ include nearby list in output
    )

    print("=== INPUTS USED ===")
    print(f"Center: ({lat}, {lng})")
    print(f"Preferred places: {preferred}")
    print()

    print("=== STRUCTURED SCHEDULE ===")
    if not day_plan["schedule"]:
        print("No stops selected. Check radius, time window, or preferred place names.")
    for s in day_plan["schedule"]:
        travel = f" • travel {s['travel_min_from_prev']} min" if "travel_min_from_prev" in s else ""
        print(f"{s['time']} • {s['place']} • score={s['score']} • crowd={s['crowd']}%{travel}")

    print("\n=== WEATHER SUMMARY ===")
    print(day_plan["weather_summary"])

    if day_plan.get("nearby_places"):
        print("\n=== NEARBY PLACES (before filtering by visit-times) ===")
        for p in day_plan["nearby_places"]:
            print(f"- {p['title']} ({p['distance_km']} km)")

    # Optional: generate a narrative using RAG
    try:
        if day_plan["schedule"]:
            stops_text = ", ".join([f"{s['time']} {s['place']}" for s in day_plan["schedule"]])
            query = (
                "Create a concise itinerary paragraph for these stops with brief highlights drawn from context. "
                "Keep place names exactly as listed and reflect expected crowd levels if relevant: "
                f"{stops_text}"
            )
            out = planner.qa_chain({"query": query})
            print("\n=== RAG NARRATIVE ===")
            print(out["result"])
        else:
            print("\n[Skipped RAG narrative: schedule is empty]")
    except Exception as e:
        print(f"\n[Skipped RAG narrative: {e}]")


if __name__ == "__main__":
    run()
