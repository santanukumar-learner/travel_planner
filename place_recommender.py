import math
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from rag import RourkelalTourismSchedulePlanner


class LatLngSchedulePlanner:
    def __init__(
        self,
        planner: "RourkelalTourismSchedulePlanner",
        default_radius_km: float = 6.0,
        dwell_minutes: int = 60,
        travel_speed_kmh: float = 20.0,
    ):
        self.p = planner
        self.default_radius_km = float(default_radius_km)
        self.dwell_minutes = int(dwell_minutes)
        self.travel_speed_kmh = float(travel_speed_kmh)

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    def _coords_for(self, key: str) -> Optional[tuple]:
        place = self.p.places_data.get(key)
        if not place:
            return None
        lat, lng = place.get("lat"), place.get("lng")
        try:
            return float(lat), float(lng)
        except Exception:
            return None

    def _travel_minutes(self, a: str, b: str) -> Optional[int]:
        ca = self._coords_for(a)
        cb = self._coords_for(b)
        if not ca or not cb:
            return None
        km = self._haversine_km(ca[0], ca[1], cb[0], cb[1])
        return int(math.ceil((km / max(1e-6, self.travel_speed_kmh)) * 60.0))

    # ✅ Weather: keep only 2-hour interval points and cap to 6 forecast points
    def _weather_for_latlng(self, lat: float, lng: float, days: int = 1) -> Dict[str, Any]:
        key = self.p.weather_api_key
        if not key:
            return {"forecast": [], "source": "none", "location": f"{lat:.3f},{lng:.3f}"}

        INTERVAL_HOURS = 2
        MAX_POINTS = 6

        try:
            url = "https://api.weatherapi.com/v1/forecast.json"
            params = {
                "key": key,
                "q": f"{lat},{lng}",
                "days": days,
                "aqi": "no",
                "alerts": "no",
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            forecast: List[Dict[str, Any]] = []
            for day in data.get("forecast", {}).get("forecastday", []):
                for hr in day.get("hour", []):
                    if len(forecast) >= MAX_POINTS:
                        break

                    ts = hr.get("time_epoch")
                    dt = datetime.fromtimestamp(ts) if ts else datetime.now()

                    # keep only 0,2,4,6,8,10,... hours
                    if dt.hour % INTERVAL_HOURS != 0:
                        continue

                    forecast.append(
                        {
                            "datetime": dt,
                            "temperature": hr.get("temp_c"),
                            "feels_like": hr.get("feelslike_c"),
                            "humidity": hr.get("humidity"),
                            "description": (hr.get("condition") or {}).get("text", "").lower(),
                            "rain": hr.get("precip_mm", 0.0),
                            "wind_speed": hr.get("wind_kph", 0.0),
                        }
                    )
                if len(forecast) >= MAX_POINTS:
                    break

            current = data.get("current", {})
            return {
                "current": {
                    "temperature": current.get("temp_c"),
                    "feels_like": current.get("feelslike_c"),
                    "humidity": current.get("humidity"),
                    "condition": (current.get("condition") or {}).get("text", "").lower()
                    if current
                    else "",
                },
                "forecast": forecast,
                "source": "weatherapi",
                "location": f"{lat:.3f},{lng:.3f}",
            }
        except Exception:
            return {"forecast": [], "source": "error", "location": f"{lat:.3f},{lng:.3f}"}

    def plan_day(
        self,
        lat: float,
        lng: float,
        date: datetime,
        radius_km: float = None,
        max_stops: int = 6,
        start_hour: int = 8,
        end_hour: int = 20,
        preferred_places: Optional[List[str]] = None,
        use_crowd: bool = False,  # will be forced based on preferred_places
        include_nearby: bool = True,
    ):
        def _norm(s: str) -> str:
            return " ".join((s or "").strip().lower().split())

        # normalize preference list
        preferred_titles = [
            _norm(p) for p in (preferred_places or [])
            if p and isinstance(p, str) and p.strip()
        ]

        # ✅ rule: if preferred places given => use_crowd=True, else False
        use_crowd = True if preferred_titles else False

        date = date or datetime.now()
        radius = float(radius_km or self.default_radius_km)

        # --- Weather summary for the day (uses capped forecast list) ---
        wx = self._weather_for_latlng(lat, lng, days=1)
        todays = [
            f for f in wx.get("forecast", [])
            if f.get("datetime") and f["datetime"].date() == date.date()
        ]
        if todays:
            temps = [f["temperature"] for f in todays if f.get("temperature") is not None]
            avg_t = sum(temps) / max(1, len(temps)) if temps else 0.0
            rain = sum(f.get("rain", 0.0) for f in todays)
            desc = todays[0].get("description", "") or ""
            weather_summary = f"{avg_t:.1f}°C, rain={rain:.1f}mm, {desc}"
        else:
            weather_summary = "No hourly forecast."

        # --- Nearby attractions (default behavior) ---
        if hasattr(self.p, "find_nearby_places"):
            nearby = self.p.find_nearby_places(
                lat, lng, radius_km=radius, kind="attraction", limit=24
            )
        else:
            nearby = []
            for item in self.p.attractions_data or []:
                plat, plng = item.get("lat"), item.get("lng")
                if plat is None or plng is None:
                    continue
                try:
                    d = self._haversine_km(float(plat), float(plng), lat, lng)
                except Exception:
                    continue
                if d <= radius:
                    nearby.append(
                        {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "lat": float(plat),
                            "lng": float(plng),
                            "distance_km": round(d, 2),
                        }
                    )
            nearby.sort(key=lambda r: r["distance_km"])
            nearby = nearby[:24]

        # ✅ If preferred places provided: ignore radius gating and only use those places
        if preferred_titles:
            all_by_title = {}
            all_by_id = {}
            for item in (self.p.attractions_data or []):
                t = _norm(item.get("title") or "")
                i = _norm(item.get("id") or "")
                if t:
                    all_by_title[t] = item
                if i:
                    all_by_id[i] = item

            selected = []
            for pref in preferred_titles:
                hit = all_by_title.get(pref) or all_by_id.get(pref)
                if not hit:
                    continue

                try:
                    plat = float(hit.get("lat")) if hit.get("lat") is not None else None
                    plng = float(hit.get("lng")) if hit.get("lng") is not None else None
                except Exception:
                    plat, plng = None, None

                selected.append(
                    {
                        "id": hit.get("id"),
                        "title": hit.get("title"),
                        "lat": plat,
                        "lng": plng,
                        "distance_km": None,  # distance is not the driver here
                    }
                )

            if selected:
                nearby = selected

        # ✅ Crowd: 2-hour interval and max 6 time slots
        CROWD_STEP_MINUTES = 120
        MAX_CROWD_SLOTS = 6
        max_end_hour_for_crowd = start_hour + (MAX_CROWD_SLOTS - 1) * (CROWD_STEP_MINUTES // 60)
        effective_end_hour = min(end_hour, max_end_hour_for_crowd)

        # --- Build candidate time slots ---
        candidates: List[Dict[str, Any]] = []

        for place in nearby:
            key = place.get("title") or place.get("id")
            if not key:
                continue

            if use_crowd:
                try:
                    recs = self.p.recommend_visit_times(
                        key,
                        date=date,
                        start_hour=start_hour,
                        end_hour=effective_end_hour,     # ✅ capped
                        step_minutes=CROWD_STEP_MINUTES, # ✅ 2-hour interval
                        top_k=2,
                    )
                except AttributeError:
                    pred = self.p.predict_crowd_level(key, date.replace(hour=start_hour))
                    recs = [
                        {
                            "place": key,
                            "time": date.replace(hour=start_hour).strftime("%Y-%m-%d %I:%M %p"),
                            "score": 60,
                            "crowd_level": pred.get("crowd_level", 50),
                            "label": pred.get("label", "Balanced"),
                            "reasons": pred.get("reasons", ["Default"]),
                        }
                    ]
            else:
                dt = date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
                base_score = 60
                recs = [
                    {
                        "place": key,
                        "time": dt.strftime("%Y-%m-%d %I:%M %p"),
                        "score": base_score,
                        "crowd_level": 50,
                        "label": "Crowd model disabled",
                        "reasons": ["Crowd prediction turned off (fast mode)"],
                    }
                ]

            for r in recs:
                dt = datetime.strptime(r["time"], "%Y-%m-%d %I:%M %p")
                candidates.append(
                    {
                        "place": r["place"],
                        "dt": dt,
                        "score": int(r["score"]),
                        "crowd": int(r["crowd_level"]),
                        "label": r.get("label", ""),
                        "reasons": r.get("reasons", []),
                    }
                )

        # --- Choose non-overlapping visits ---
        candidates.sort(key=lambda x: (x["dt"], -x["score"]))
        chosen: List[Dict[str, Any]] = []

        for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
            if len(chosen) >= max_stops:
                break
            if not chosen:
                chosen.append(c)
                continue
            prev = chosen[-1]
            tm = self._travel_minutes(prev["place"], c["place"]) or 0
            gap = (c["dt"] - prev["dt"]).total_seconds() / 60.0
            if gap >= (self.dwell_minutes + tm):
                chosen.append(c)

        # --- Order by proximity if available ---
        if hasattr(self.p, "order_stops_by_proximity"):
            ordered = self.p.order_stops_by_proximity([x["place"] for x in chosen])
            order_map = {o["title"]: i for i, o in enumerate(ordered, 1)}
            chosen.sort(key=lambda x: (order_map.get(x["place"], 999), x["dt"]))

        # --- Final schedule ---
        schedule: List[Dict[str, Any]] = []
        for i, c in enumerate(chosen, 1):
            item = {
                "order": i,
                "time": c["dt"].strftime("%I:%M %p"),
                "place": c["place"],
                "score": c["score"],
                "crowd": c["crowd"],
                "note": c["label"] or ", ".join(c["reasons"]) or "Good trade-off",
            }
            if i > 1:
                tm = self._travel_minutes(chosen[i - 2]["place"], c["place"])
                if tm is not None:
                    item["travel_min_from_prev"] = tm
            schedule.append(item)

        result: Dict[str, Any] = {
            "date": date.strftime("%Y-%m-%d"),
            "center": {"lat": lat, "lng": lng},
            "weather_summary": weather_summary,
            "schedule": schedule,
        }
        if include_nearby:
            result["nearby_places"] = nearby
        return result


if __name__ == "__main__":
    planner = RourkelalTourismSchedulePlanner(
        api_key=os.getenv("GEMINI_API_KEY"),
        weather_api_key=os.getenv("WHEATHER_KEY"),
        attractions_file="rag.json",
        restaurants_file="restaurants_rourkela.json",
        model_file="rkl_places_xgb.pkl",
    )

    geo = LatLngSchedulePlanner(
        planner, default_radius_km=6.0, dwell_minutes=60, travel_speed_kmh=20.0
    )

    lat, lng = 22.230, 84.826
    day_plan = geo.plan_day(
        lat=lat,
        lng=lng,
        date=datetime.now(),
        radius_km=6.0,                  # used only when preferred_places is empty
        max_stops=4,
        preferred_places=["YOUR PLACE TITLE HERE"],  # if provided -> crowd auto True
    )

    from pprint import pprint
    pprint(day_plan)
