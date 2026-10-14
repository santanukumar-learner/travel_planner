"""Local, key-free itinerary planning for the Rourkela place dataset."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path


DATA_FILE = Path(__file__).with_name("rag.json")


def load_places():
    with DATA_FILE.open(encoding="utf-8") as handle:
        places = json.load(handle)
    return [p for p in places if p.get("lat") is not None and p.get("lng") is not None]


def distance_km(a_lat, a_lng, b_lat, b_lng):
    radius = 6371.0
    lat1, lat2 = math.radians(a_lat), math.radians(b_lat)
    delta_lat = lat2 - lat1
    delta_lng = math.radians(b_lng - a_lng)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def plan_day(places, center, day, start_hour, end_hour, radius_km, max_stops, selected_titles):
    """Choose nearby stops, then order them by a simple nearest-neighbor route."""
    if end_hour <= start_hour:
        raise ValueError("End time must be after start time.")
    selected = set(selected_titles)
    nearby = []
    for place in places:
        dist = distance_km(center[0], center[1], float(place["lat"]), float(place["lng"]))
        if dist <= radius_km or place["title"] in selected:
            nearby.append({**place, "distance_from_start_km": round(dist, 2)})
    nearby.sort(key=lambda p: (p["title"] not in selected, p["distance_from_start_km"]))
    pool = nearby[:max_stops] if not selected else [p for p in nearby if p["title"] in selected][:max_stops]

    current = center
    now = datetime.combine(day, datetime.min.time()).replace(hour=start_hour)
    deadline = now.replace(hour=end_hour)
    itinerary = []
    while pool:
        place = min(pool, key=lambda p: distance_km(current[0], current[1], float(p["lat"]), float(p["lng"])))
        pool.remove(place)
        leg = distance_km(current[0], current[1], float(place["lat"]), float(place["lng"]))
        travel_minutes = round(leg / 20 * 60)
        arrival = now + timedelta(minutes=travel_minutes)
        departure = arrival + timedelta(minutes=60)
        if departure > deadline:
            break
        itinerary.append({
            "order": len(itinerary) + 1,
            "time": arrival.strftime("%I:%M %p"),
            "place": place["title"],
            "distance_km": round(leg, 2),
            "travel_minutes": travel_minutes,
            "lat": float(place["lat"]),
            "lng": float(place["lng"]),
            "description": place.get("content", ""),
        })
        now, current = departure, (float(place["lat"]), float(place["lng"]))
    return nearby, itinerary
