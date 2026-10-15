import math
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
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
        d = self._haversine_km(ca[0], ca[1], cb[0], cb[1])
        hours = d / max(1e-6, self.travel_speed_kmh)
        return int(round(hours * 60))

    def _weather_for_latlng(self, lat: float, lng: float, days: int = 1) -> Dict[str, Any]:
        key = self.p.weather_api_key
        if not key:
            return {"forecast": [], "source": "none", "location": f"{lat:.3f},{lng:.3f}"}

        INTERVAL_HOURS = 2
        MAX_POINTS = 6

        try:
            url = "https://api.weatherapi.com/v1/forecast.json"
            params = {"key": key, "q": f"{lat},{lng}", "days": days, "aqi": "no", "alerts": "no"}
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
                    "condition": (current.get("condition") or {}).get("text", "").lower() if current else "",
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
        use_crowd: bool = False,
        include_nearby: bool = True,
    ):
        def _norm(s: str) -> str:
            return " ".join((s or "").strip().lower().split())

        preferred_list: List[str] = []
        seen = set()
        for p in (preferred_places or []):
            if not p or not isinstance(p, str):
                continue
            n = _norm(p)
            if n and n not in seen:
                preferred_list.append(n)
                seen.add(n)
        preferred_set = set(preferred_list)
        use_crowd = True if preferred_list else False

        date = date or datetime.now()
        radius = float(radius_km or self.default_radius_km)

        wx = self._weather_for_latlng(lat, lng, days=1)
        todays = [f for f in (wx.get("forecast") or []) if isinstance(f.get("datetime"), datetime)]
        if todays:
            avg_temp = sum((f.get("temperature") or 0.0) for f in todays) / max(1, len(todays))
            total_rain = sum((f.get("rain") or 0.0) for f in todays)
            weather_summary = f"{avg_temp:.1f}°C avg, rain {total_rain:.1f}mm"
        else:
            weather_summary = "No forecast."

        nearby: List[Dict[str, Any]] = []

        if not preferred_list:
            for item in (self.p.attractions_data or []):
                try:
                    plat = float(item.get("lat")) if item.get("lat") is not None else None
                    plng = float(item.get("lng")) if item.get("lng") is not None else None
                except Exception:
                    plat, plng = None, None
                if plat is None or plng is None:
                    continue
                d = self._haversine_km(lat, lng, plat, plng)
                if d <= radius:
                    nearby.append(
                        {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "lat": plat,
                            "lng": plng,
                            "distance_km": round(d, 2),
                        }
                    )
            nearby.sort(key=lambda r: r["distance_km"])
            nearby = nearby[:24]

        if preferred_list:
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
            for pref in preferred_list:
                hit = all_by_title.get(pref) or all_by_id.get(pref)
                if not hit:
                    continue

                # ✅ extract lat/lng safely
                try:
                    plat = float(hit.get("lat")) if hit.get("lat") is not None else None
                    plng = float(hit.get("lng")) if hit.get("lng") is not None else None
                except Exception:
                    plat, plng = None, None

                # ✅ compute distance from current center (lat,lng passed to plan_day)
                if plat is not None and plng is not None:
                    d0 = self._haversine_km(lat, lng, plat, plng)
                    distance_km = round(d0, 2)
                else:
                    distance_km = None

                selected.append(
                    {
                        "id": hit.get("id"),
                        "title": hit.get("title"),
                        "lat": plat,
                        "lng": plng,
                        "distance_km": distance_km,
                    }
                )

            # ✅ after loop
            if selected:
                nearby = selected

        CROWD_STEP_MINUTES = 120
        MAX_CROWD_SLOTS = 6
        max_end_hour_for_crowd = start_hour + (MAX_CROWD_SLOTS - 1) * (CROWD_STEP_MINUTES // 60)
        effective_end_hour = min(end_hour, max_end_hour_for_crowd)

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
                        end_hour=effective_end_hour,
                        step_minutes=CROWD_STEP_MINUTES,
                        top_k=2,
                    )
                except AttributeError:
                    pred = self.p.predict_crowd_level(key, date.replace(hour=start_hour))
                    recs = [
                        {
                            "place": key,
                            "time": date.replace(hour=start_hour).strftime("%Y-%m-%d %I:%M %p"),
                            "score": 60,
                            "crowd_level": pred.get("crowd_level", 60),
                        }
                    ]
            else:
                dist = place.get("distance_km", 9.9) or 9.9
                base = max(0, int(round(100 - dist * 8)))
                dt = date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
                recs = [
                    {
                        "place": place.get("title"),
                        "time": dt.strftime("%Y-%m-%d %I:%M %p"),
                        "score": base,
                        "crowd_level": place.get("crowd", 50) or 50,
                    }
                ]

            for r in recs:
                dt = datetime.strptime(r["time"], "%Y-%m-%d %I:%M %p")
                candidates.append(
                    {
                        "place": r["place"],
                        "dt": dt,
                        "score": int(r.get("score", 0)),
                        "crowd": int(r.get("crowd_level", 0)),
                        "label": r.get("label", ""),
                        "reasons": r.get("reasons", []),
                    }
                )

        candidates.sort(key=lambda x: (x["dt"], -x["score"]))

        def fits(candidate: Dict[str, Any], chosen_list: List[Dict[str, Any]]) -> bool:
            for prev in chosen_list:
                tm = self._travel_minutes(prev["place"], candidate["place"]) or 0
                gap = abs((candidate["dt"] - prev["dt"]).total_seconds()) / 60.0
                if gap < (self.dwell_minutes + tm):
                    return False
            return True

        chosen: List[Dict[str, Any]] = []
        chosen_place_norms = set()

        if preferred_list:
            by_place: Dict[str, List[Dict[str, Any]]] = {}
            for c in candidates:
                by_place.setdefault(_norm(c["place"]), []).append(c)

            for pref in preferred_list:
                opts = by_place.get(pref, [])
                opts.sort(key=lambda x: x["score"], reverse=True)
                picked = None
                for c in opts:
                    if fits(c, chosen):
                        picked = c
                        break
                if picked is not None:
                    chosen.append(picked)
                    chosen_place_norms.add(_norm(picked["place"]))

        for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
            if len(chosen) >= max_stops:
                break
            if _norm(c["place"]) in chosen_place_norms:
                continue
            if fits(c, chosen):
                chosen.append(c)
                chosen_place_norms.add(_norm(c["place"]))

        chosen.sort(key=lambda x: x["dt"])

        schedule: List[Dict[str, Any]] = []
        for i, c in enumerate(chosen, 1):
            item = {
                "order": i,
                "time": c["dt"].strftime("%I:%M %p"),
                "place": c["place"],
                "score": c["score"],
                "crowd": c["crowd"],
                "note": c["label"] if c["label"] else "",
            }
            if i > 1:
                prev_place = chosen[i - 2]["place"]
                tm = self._travel_minutes(prev_place, c["place"])
                if tm is not None:
                    item["travel_min_from_prev"] = tm

                # dynamic distance from prev
                ca = self._coords_for(prev_place)
                cb = self._coords_for(c["place"])
                if ca and cb:
                    item["distance_from_prev_km"] = round(self._haversine_km(ca[0], ca[1], cb[0], cb[1]), 2)

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
