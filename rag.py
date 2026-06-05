import json
import os
import math
import requests
import pickle
import calendar
import numpy as np

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv

from data_converter import CrowdPredictionInputProcessor
from pathway_retriever_client import PathwayRetrieverClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class RourkelalTourismSchedulePlanner:
    def __init__(
        self,
        
        api_key: str,
        weather_api_key: str = None,
        attractions_file: str = "rag.json",
        restaurants_file: str = "restaurants_rourkela.json",
        model_file: str = "rkl_places_xgb.pkl",
    ):

        load_dotenv()
        env_api = os.getenv("GEMINI_API_KEY")
        env_weather = os.getenv("WHEATHER_KEY")

        # explicit args win; fall back to env
        self.api_key = api_key or env_api
        self.weather_api_key = weather_api_key or env_weather
        if not self.api_key:
            raise ValueError("Missing Gemini API key (api_key or GEMINI_API_KEY)")
        # weather key optional; we’ll fallback to sample weather if absent

        # normalize paths relative to this file
        base = Path(__file__).resolve().parent
        self.attractions_file = str((base / attractions_file).resolve())
        self.restaurants_file = str((base / restaurants_file).resolve())
        self.model_file = str((base / model_file).resolve())

        self.city_lat = 22.2396
        self.city_lng = 84.8633
        self.processor = CrowdPredictionInputProcessor()
        self._weather_cache = {}  # key: (date, hour) -> {"temp":..., "rain_flag":...}
        self.attractions_data: List[Dict[str, Any]] = []
        self.restaurants_data: List[Dict[str, Any]] = []
        self.crowd_predictions: Dict[str, Any] = {}
        self.places_data: Dict[str, Any] = {}

        self._load_data()
        self._load_crowd_model()
        self._build_place_id_map()

        # embeddings + LLM
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=self.api_key,
        )

        self._setup_vectorstore()
        self._setup_qa_chain()

    def _build_place_id_map(self):
        """Allow lookup by either id or title."""
        self.place_id_map: Dict[str, str] = {}
        for item in self.attractions_data:
            pid = item.get("id")
            title = item.get("title")
            if pid and title:
                self.place_id_map[pid] = title
                self.place_id_map[title] = title  # title→title for normalization

    def _load_data(self):
        self.attractions_data, self.restaurants_data = [], []
        # --- attractions ---
        try:
            with open(self.attractions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.attractions_data = data
            elif isinstance(data, dict) and "attractions" in data:
                self.attractions_data = data["attractions"]
        except FileNotFoundError:
            print(f"⚠️ Attractions file not found: {self.attractions_file}")
        except Exception as e:
            print(f"⚠️ Attractions load error: {e}")

        # --- restaurants ---
        try:
            with open(self.restaurants_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.restaurants_data = data if isinstance(data, list) else []
        except FileNotFoundError:
            print(f"⚠️ Restaurants file not found: {self.restaurants_file}")
        except Exception as e:
            print(f"⚠️ Restaurants load error: {e}")

        self._build_places_data()
        print(
            f"✅ Loaded {len(self.attractions_data)} attractions, "
            f"{len(self.restaurants_data)} restaurants"
        )

    def _build_places_data(self):
        """Build dictionary of places keyed by both ID and Title."""
        self.places_data = {}

        if not hasattr(self, "attractions_data"):
            print("⚠️ No attractions data found before building places_data")
            return

        for item in self.attractions_data:
            place_id = item.get("id")
            title = item.get("title")

            if place_id:
                self.places_data[place_id] = item
            if title:
                self.places_data[title] = item

        print(
            f"✅ Built places_data with {len(self.places_data)} entries. "
            f"Examples: {list(self.places_data.keys())[:5]}"
        )

    def _load_crowd_model(self):
        try:
            with open(self.model_file, "rb") as f:
                self.crowd_model = pickle.load(f)
            if not hasattr(self.crowd_model, "predict"):
                raise TypeError("Loaded model has no .predict(...) method")
            print("✅ Crowd prediction model loaded")
        except FileNotFoundError:
            raise FileNotFoundError(f"Crowd model file not found: {self.model_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_file}: {e}")

    # ------------------------------------------------------------------ #
    # Weather
    # ------------------------------------------------------------------ #
    def get_weather_forecast(self, days: int = 3) -> Dict[str, Any]:
        return self.get_current_weather(api_key=self.weather_api_key, days=days)

    def get_current_weather(self, api_key=None, days: int = 3) -> Dict[str, Any]:
        """Get current weather + forecast. Falls back to synthetic sample if API fails/missing."""
        if api_key:
            try:
                # current
                url_current = "https://api.weatherapi.com/v1/current.json"
                params_current = {
                    "key": api_key,
                    "q": f"{self.city_lat},{self.city_lng}",
                    "aqi": "no",
                }
                resp_current = requests.get(
                    url_current, params=params_current, timeout=10
                )
                resp_current.raise_for_status()
                data_current = resp_current.json()
                cur = data_current.get("current", {})
                condition_text = (cur.get("condition") or {}).get("text", "") or ""
                condition = condition_text.lower()

                rain_terms = {
                    "rain",
                    "drizzle",
                    "shower",
                    "thunderstorm",
                    "storm",
                    "precipitation",
                    "wet",
                    "pour",
                }
                rain_flag = (
                    1
                    if (cur.get("precip_mm", 0) or 0) > 0
                    or any(t in condition for t in rain_terms)
                    else 0
                )

                current = {
                    "temperature": cur.get("temp_c", 28.0),
                    "feels_like": cur.get("feelslike_c", cur.get("temp_c", 28.0)),
                    "humidity": cur.get("humidity", 60),
                    "condition": condition,
                    "rain_flag": rain_flag,
                    "wind_speed": cur.get("wind_kph", 0.0),
                }

                # forecast
                forecast = []
                try:
                    url_forecast = "https://api.weatherapi.com/v1/forecast.json"
                    params_forecast = {
                        "key": api_key,
                        "q": f"{self.city_lat},{self.city_lng}",
                        "days": days,
                        "aqi": "no",
                        "alerts": "no",
                    }
                    resp_forecast = requests.get(
                        url_forecast, params=params_forecast, timeout=10
                    )
                    resp_forecast.raise_for_status()
                    data_forecast = resp_forecast.json()
                    for day in data_forecast.get("forecast", {}).get(
                        "forecastday", []
                    ):
                        for hour in day.get("hour", []):
                            ts = hour.get("time_epoch")
                            dt = datetime.fromtimestamp(ts) if ts else None
                            forecast.append(
                                {
                                    "datetime": dt or datetime.now(),
                                    "temperature": hour.get(
                                        "temp_c", current["temperature"]
                                    ),
                                    "feels_like": hour.get(
                                        "feelslike_c", current["feels_like"]
                                    ),
                                    "humidity": hour.get(
                                        "humidity", current["humidity"]
                                    ),
                                    "description": (
                                        (hour.get("condition") or {})
                                        .get("text", "")
                                        .lower()
                                    ),
                                    "rain": hour.get("precip_mm", 0.0),
                                    "wind_speed": hour.get("wind_kph", 0.0),
                                }
                            )
                except Exception as fe:
                    print(f"⚠️ Forecast error: {fe}")
                    forecast = self._generate_sample_weather(days)["forecast"]

                return {
                    "current": current,
                    "forecast": forecast,
                    "source": "weatherapi",
                    "location": f"Rourkela ({self.city_lat}, {self.city_lng})",
                }

            except Exception as e:
                return self._generate_sample_weather(days)
        else:
            print("⚠️ No WEATHER_KEY. Using sample weather.")
            return self._generate_sample_weather(days)

    def _generate_sample_weather(self, days: int) -> Dict[str, Any]:
        current_date = datetime.now()
        forecast = []

        for day in range(days):
            date = current_date + timedelta(days=day)
            # Seasonal ranges
            month = date.month
            if month in [12, 1, 2]:  # Winter
                temp_range = (15, 25)
                rain_chance = 0.1
            elif month in [3, 4, 5]:  # Summer
                temp_range = (25, 40)
                rain_chance = 0.2
            elif month in [6, 7, 8, 9]:  # Monsoon
                temp_range = (20, 32)
                rain_chance = 0.7
            else:  # Post-monsoon
                temp_range = (18, 30)
                rain_chance = 0.3

            # Generate 3-hourly forecast
            for hour in range(0, 24, 3):
                forecast_time = date.replace(hour=hour, minute=0, second=0)
                temp = temp_range[0] + (temp_range[1] - temp_range[0]) * (
                    0.5 + 0.5 * math.sin(math.pi * hour / 12)
                )

                forecast.append(
                    {
                        "datetime": forecast_time,
                        "temperature": round(temp, 1),
                        "feels_like": round(temp + 2, 1),
                        "humidity": 60 + (rain_chance * 30),
                        "description": (
                            "rainy"
                            if rain_chance > 0.5
                            else "clear"
                            if rain_chance < 0.2
                            else "cloudy"
                        ),
                        "rain": 5.0 if rain_chance > 0.6 else 0.0,
                        "wind_speed": 5.0,
                    }
                )

        return {
            "forecast": forecast,
            "source": "sample_data",
            "location": "Rourkela (Sample Data)",
        }


    def predict_crowd_level(
        self, place_name: str, visit_datetime: datetime, weather_data: Dict = None
    ) -> Dict[str, Any]:
        """
        Predict crowd level for a specific place and time.
        
        Args:
            place_name: Name or ID of the place
            visit_datetime: When to visit
            weather_data: Optional weather data (unused, kept for compatibility)
        
        Returns:
            Dictionary with crowd prediction and recommendations
        """
        # Normalize place (allow id or title)
        if place_name in self.places_data:
            place = self.places_data[place_name]
            place_name = place.get("title", place_name)
        else:
            raise ValueError(
                f"Place '{place_name}' not found. "
                f"Available places: {list(self.place_id_map.values())[:5]}..."
            )

        model = self.crowd_model

        # Prepare features
        feature_vector = self.processor.prepare_model_input(
            place_name=place_name,
            target_datetime=visit_datetime,
            target_hour=visit_datetime.hour,
            weather_api_key=self.weather_api_key,
        )
        X = np.array(feature_vector, dtype=float).reshape(1, -1)

        # ✅ CHANGE 1: SIMPLIFIED - XGBRegressor predicts crowd % directly
        y = float(model.predict(X)[0])
        crowd_level = int(max(0, min(120, round(y))))
        
        # ✅ CHANGE 2: Better confidence calculation based on training range
        if crowd_level <= 95:
            confidence = "high"      # Normal range
        elif crowd_level <= 110:
            confidence = "medium"    # Holiday range
        else:
            confidence = "low"       # Outside typical range
        
        # ✅ CHANGE 3: More descriptive messages
        if crowd_level < 30:
            desc = f"Low crowd expected (~{crowd_level}%) - Great time to visit!"
        elif crowd_level < 60:
            desc = f"Moderate crowd expected (~{crowd_level}%) - Good visiting conditions"
        elif crowd_level < 90:
            desc = f"High crowd expected (~{crowd_level}%) - Consider alternative times"
        else:
            desc = f"Very high crowd expected (~{crowd_level}%) - Peak time, plan accordingly"

        # ✅ CHANGE 4: Improved alternative time finding
        best_alternatives: List[str] = []
        
        for hour_shift in [-2, -1, 1, 2]:
            alt_time = visit_datetime + timedelta(hours=hour_shift)
            
            # ✅ CHANGE 5: Skip times outside operating hours
            if not (7 <= alt_time.hour <= 20):
                continue
            
            try:
                alt_features = self.processor.prepare_model_input(
                    place_name=place_name,
                    target_datetime=alt_time,
                    target_hour=alt_time.hour,
                    weather_api_key=self.weather_api_key,
                )
                X_alt = np.array(alt_features, dtype=float).reshape(1, -1)
                
                # ✅ CHANGE 6: SIMPLIFIED - Direct prediction
                y_alt = float(model.predict(X_alt)[0])
                alt_level = int(max(0, min(120, round(y_alt))))

                # ✅ CHANGE 7: Only suggest if significantly better (10% threshold)
                if alt_level < (crowd_level - 10):
                    best_alternatives.append(
                        f"{alt_time.strftime('%I:%M %p')} (~{alt_level}% crowd)"
                    )
            except Exception:
                continue

        # ✅ CHANGE 8: Better default suggestions
        if not best_alternatives:
            if crowd_level > 60:
                best_alternatives = [
                    "Early morning (7-9 AM) typically has lower crowds",
                    "Late evening (after 6 PM) is often less busy"
                ]
            else:
                best_alternatives = ["Current time is already good - no better alternatives"]

        # ✅ CHANGE 9: Error handling for context
        try:
            context = self.processor.get_prediction_context(
                place_name=place_name,
                target_datetime=visit_datetime,
                target_hour=visit_datetime.hour,
                weather_api_key=self.weather_api_key,
            )
        except Exception as e:
            context = {
                "place": place_name,
                "datetime": visit_datetime.isoformat(),
                "error": str(e)
            }

        return {
            "place": place_name,
            "datetime": visit_datetime.isoformat(),
            "crowd_level": crowd_level,
            "description": desc,
            "probability": min(1.0, crowd_level / 100.0),  # Normalize to 0-1
            "context": context,
            "confidence": confidence,
            "best_alternative_times": best_alternatives[:3],  # ✅ CHANGE 10: Limit to top 3
        }
    def find_nearby_places(
        self,
        center_lat: float,
        center_lng: float,
        radius_km: float = 6.0,
        kind: str | None = None,
        limit: int = 10,
    ):
        """
        Return a list of nearby attractions within radius_km, sorted by distance.
        kind: "attraction" or "restaurant" to filter via metadata['type'] (optional).
        """
        results = []
        for item in self.attractions_data:
            lat, lng = item.get("lat"), item.get("lng")
            if lat is None or lng is None:
                continue
            try:
                dist = haversine_km(float(lat), float(lng), center_lat, center_lng)
            except Exception:
                continue
            if dist <= radius_km:
                if kind and kind != "attraction":
                    continue
                results.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "lat": float(lat),
                        "lng": float(lng),
                        "distance_km": round(dist, 2),
                    }
                )
        results.sort(key=lambda r: r["distance_km"])
        return results[:limit]

    def order_stops_by_proximity(
        self,
        place_keys: list[str],
        start_lat: float | None = None,
        start_lng: float | None = None,
    ):
        """Greedy nearest-neighbor ordering of a small list of places."""
        pts = []
        for k in place_keys:
            coords = self._get_place_coords(k)
            if coords:
                lat, lng = coords
                title = self.place_id_map.get(k, k)
                pts.append({"key": k, "title": title, "lat": lat, "lng": lng})
        if not pts:
            return []

        cur_lat = start_lat if start_lat is not None else self.city_lat
        cur_lng = start_lng if start_lng is not None else self.city_lng
        remaining = pts[:]
        route = []
        while remaining:
            nxt = min(
                remaining,
                key=lambda p: haversine_km(cur_lat, cur_lng, p["lat"], p["lng"]),
            )
            route.append(nxt)
            cur_lat, cur_lng = nxt["lat"], nxt["lng"]
            remaining.remove(nxt)
        return route

    def _get_place_coords(self, key: str):
        """key can be id ('religious_1') or title ('Hanuman Vatika')."""
        place = self.places_data.get(key)
        if not place:
            return None
        lat, lng = place.get("lat"), place.get("lng")
        try:
            return float(lat), float(lng)
        except (TypeError, ValueError):
            return None

    def _suggest_better_times(self, place_id: str, current_time: datetime) -> List[str]:
        """Suggest better times to visit if current time has high crowds"""
        suggestions = []
        crowd_data = self.crowd_predictions.get(place_id, {})

        for hour in [7, 9, 11, 15, 17, 19]:
            test_time = current_time.replace(hour=hour)
            test_crowd = self.predict_crowd_level(place_id, test_time)
            if test_crowd["crowd_level"] < 50:
                time_str = test_time.strftime("%I:%M %p")
                suggestions.append(
                    f"{time_str} (Crowd: {test_crowd['crowd_level']}%)"
                )

        return suggestions[:3]

    def _setup_vectorstore(self):
        import shutil

        idx_dir = "faiss_index"

        # Try load first (fast path)
        try:
            self.vectorstore = FAISS.load_local(
                idx_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"✅ Loaded FAISS index from '{idx_dir}'")
            return
        except Exception:
            pass

        # Build docs only if load fails
        documents: List[Document] = []

        for item in self.attractions_data:
            pid, title = item.get("id"), item.get("title")
            if not pid or not title:
                continue
            crowd_info = self.crowd_predictions.get(pid, {})
            content = f"""
    Type: Tourist Attraction - {self._categorize_place(pid)}
    Name: {title}
    Description: {item.get('content','')}
    Location: Latitude {item.get('lat','N/A')}, Longitude {item.get('lng','N/A')}
    Category: {self._categorize_place(pid)}
    Best for: {self._get_best_for_category(pid)}

    Crowd Information:
    - Busiest days: {self._get_busiest_days(crowd_info)}
    - Best times to visit: Early morning (6-8 AM) or late evening (after 7 PM)
    - Weather sensitivity: {'High' if crowd_info.get('weather_sensitivity', {}).get('rain', 0) >= 0.5 else 'Low'}
    - Capacity level: {crowd_info.get('capacity_level', 'medium')}
    """.strip()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": pid,
                        "name": title,
                        "type": "attraction",
                        "category": self._categorize_place(pid),
                        "lat": item.get("lat"),
                        "lng": item.get("lng"),
                        "has_crowd_data": pid in self.crowd_predictions,
                        "weather_sensitive": crowd_info.get("weather_sensitivity", {}).get("rain", 0) >= 0.5,
                    },
                )
            )

        for item in self.restaurants_data:
            name = item.get("name")
            if not name:
                continue
            rating = item.get("rating") or 0
            rating_count = item.get("user_ratings_total") or 0
            content = f"""
    Type: Restaurant
    Name: {name}
    Address: {item.get('address','')}
    Rating: {rating}/5 ({rating_count} reviews)
    Location: Latitude {item.get('lat','')}, Longitude {item.get('lng','')}
    Cuisine: {self._guess_cuisine_type(name)}
    Price Range: {self._estimate_price_range(name, rating)}
    Good for: {self._get_restaurant_suitable_for(name, rating)}

    Crowd & Timing:
    - Peak hours: 12-2 PM (lunch), 7-9 PM (dinner)
    - Less crowded: 3-6 PM, after 9 PM
    - Weather impact: Indoor dining largely unaffected
    - Reservation recommended: {'Yes' if rating > 4.0 else 'Not required'}
    """.strip()
            

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": item.get("place_id", name),
                        "name": name,
                        "type": "restaurant",
                        "category": "dining",
                        "lat": item.get("lat"),
                        "lng": item.get("lng"),
                        "rating": rating,
                        "rating_count": rating_count,
                        "price_level": self._estimate_price_range(name, rating),
                    },
                )
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Rebuild index
        if os.path.exists(idx_dir):
            shutil.rmtree(idx_dir, ignore_errors=True)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(idx_dir)
        print(f"✅ Created FAISS index with {len(chunks)} chunks → '{idx_dir}'")


    def _setup_qa_chain(self):
        allowed_titles = sorted(
            {a["title"] for a in self.attractions_data if a.get("title")}
        )
        allowed_block = "\n".join(f"- {t}" for t in allowed_titles)

        prompt_template = """
You are an expert Rourkela trip planner. Use ONLY the provided context and the allowlist of places.

ALLOWLIST (you MUST ONLY reference places from this list; do not invent new names):
{allowed_places}
Context:
{context}

Question:
{question}

IMPORTANT PLANNING CONSIDERATIONS:
Weather & Timing:
- Check weather conditions before recommending outdoor activities
- Suggest indoor alternatives for rainy days
- Consider temperature comfort (ideal: 15-30°C)
- Avoid midday visits (12-3 PM) during hot weather (>35°C)

Crowd Management:
- Suggest weekdays over weekends for better experience
- Provide alternative timings when places are less crowded
- Factor in festival seasons and special events

Practical Schedule Planning:
- Group nearby locations to minimize travel time
- Include buffer time for unexpected delays
- Suggest meal times aligned with restaurant peak hours
- Consider opening/closing times of attractions
- Include rest breaks, especially for families

Weather-Specific Recommendations:
- Rainy Day: Indoor attractions (temples, museums, covered markets), shopping
- Hot Weather: Early morning visits, air-conditioned restaurants, parks with shade
- Pleasant Weather: Outdoor activities, dam visits, longer walks

Provide detailed schedules with:
1. Specific timings with weather considerations
2. Crowd level expectations and alternatives
4. Backup plans for weather changes
5. Restaurant timing to avoid peak crowds
6. Transportation tips considering weather

Answer:
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            partial_variables={"allowed_places": allowed_block},
        )

        # 🔹 Pathway client (replaces FAISS)
        pathway_client = PathwayRetrieverClient("http://127.0.0.1:8765")

        def pathway_retrieve(question: str) -> str:
            results = pathway_client.search(question, k=12)

            # same format_docs behavior, but manual
            rag_context = "\n\n".join(r["text"] for r in results)
            return rag_context


        self.qa_chain = (
            {
                "question": RunnablePassthrough(),
                "context": pathway_retrieve,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )


    # ------------------------------------------------------------------ #
    # Smart schedule & recommendations
    # ------------------------------------------------------------------ #
    def create_smart_schedule(
        self,
        duration_days: int,
        interests: List[str] = None,
        budget: str = "moderate",
        group_type: str = "family",
        start_date: datetime = None,
    ) -> Dict[str, Any]:
        duration_days = int(duration_days or 1)
        if duration_days < 1:
            duration_days = 1
        if duration_days > 7:
            duration_days = 7
        interests = interests or ["religious", "nature", "food"]
        start_date = start_date or datetime.now()

        weather_forecast = self.get_weather_forecast(duration_days)
        daily_weather: Dict[int, Dict[str, Any]] = {}
        if weather_forecast.get("forecast"):
            for day in range(duration_days):
                day_date = start_date + timedelta(days=day)
                day_forecasts = [
                    f
                    for f in weather_forecast["forecast"]
                    if f["datetime"].date() == day_date.date()
                ]
                if day_forecasts:
                    avg_temp = sum(
                        f["temperature"] for f in day_forecasts
                    ) / len(day_forecasts)
                    total_rain = sum(f.get("rain", 0.0) for f in day_forecasts)
                    daily_weather[day] = {
                        "date": day_date,
                        "avg_temperature": avg_temp,
                        "rain": total_rain,
                        "description": day_forecasts[0].get(
                            "description", "clear"
                        )
                        or "clear",
                    }

        weather_context = self._create_weather_context(daily_weather)

        query = f"""
Create a detailed {duration_days}-day tourist schedule for Rourkela starting from {start_date.strftime('%Y-%m-%d')} with the following preferences:

PREFERENCES:
- Interests: {', '.join(interests)}
- Budget: {budget}
- Group type: {group_type}

WEATHER FORECAST:
{weather_context}

REQUIREMENTS:
- Include specific timings optimized for weather and crowds
- Suggest indoor alternatives for rainy days
- Provide crowd level expectations for each attraction
- Include restaurant recommendations with timing
- Consider travel time between locations
- Add weather-appropriate tips and alternatives
- Optimize for minimal travel time and maximum experience
"""

        try:
            answer = self.qa_chain.invoke(query)
            sources_used = None  # LCEL chain not exposing docs here
        except Exception as e:
            answer = f"(Planner fallback) {e}"
            sources_used = None

        schedule_with_predictions = self._enhance_schedule_with_predictions(
            answer, daily_weather
        )

        return {
            "schedule": schedule_with_predictions,
            "weather_forecast": daily_weather,
            "duration_days": duration_days,
            "interests": interests,
            "budget": budget,
            "group_type": group_type,
            "sources_used": sources_used,
        }
    
    def build_day_schedule_from_center(
        self,
        date: datetime,
        center_lat: float,
        center_lng: float,
        preferred_titles: list[str] | None = None,
        radius_km: float = 40.0,
        max_stops: int = 4,
    ):
        # 1️⃣ Compute nearby places WITH distance
        nearby = self.find_nearby_places(
            center_lat=center_lat,
            center_lng=center_lng,
            radius_km=radius_km,
            limit=12,
        )

        # 2️⃣ Ensure preferred places are included
        preferred_titles = preferred_titles or []
        stops = []

        for t in preferred_titles:
            if t in self.places_data:
                stops.append(t)

        for p in nearby:
            if p["title"] not in stops:
                stops.append(p["title"])
            if len(stops) >= max_stops:
                break

        # 3️⃣ Assign best visit times
        items = []
        used_hours = set()
        base_day = date.replace(hour=0, minute=0, second=0, microsecond=0)

        for place in stops:
            best = self.best_visit_time(place, base_day)
            if not best:
                continue

            visit_dt = datetime.strptime(best["time"], "%Y-%m-%d %I:%M %p")

            # avoid same-hour collision
            while visit_dt.hour in used_hours and visit_dt.hour < 20:
                visit_dt += timedelta(hours=1)

            used_hours.add(visit_dt.hour)

            items.append({
                "order": len(items) + 1,
                "time": visit_dt.strftime("%I:%M %p"),
                "place": place,
                "score": best["score"],
                "crowd": best["crowd_level"],
                "note": "High crowd" if best["crowd_level"] > 60 else "Good time",
            })

        return {
            "schedule": items,
            "nearby_places": nearby,  # distances INCLUDED
        }
    def build_day_schedule_dynamic_route(
        self,
        date: datetime,
        start_lat: float,
        start_lng: float,
        preferred_titles: list[str] | None = None,
        radius_km: float = 40.0,
        max_stops: int = 4,
        include_debug_nearby: bool = False,
    ):
        """
        Build a schedule where after each visited place, the center updates to that place
        and distances are recalculated for the next selection.
        """

        preferred_titles = preferred_titles or []

        # Normalize preferred: only keep valid places
        preferred_queue = []
        for t in preferred_titles:
            if t in self.places_data:
                preferred_queue.append(self.places_data[t].get("title", t))
            elif t in self.place_id_map and self.place_id_map[t] in self.places_data:
                preferred_queue.append(self.place_id_map[t])

        base_day = date.replace(hour=0, minute=0, second=0, microsecond=0)

        current_lat, current_lng = float(start_lat), float(start_lng)
        visited = set()
        schedule_items = []
        debug_nearby_snapshots = []

        def _dist_to(place_title: str, from_lat: float, from_lng: float) -> float | None:
            coords = self._get_place_coords(place_title)
            if not coords:
                return None
            return round(haversine_km(from_lat, from_lng, coords[0], coords[1]), 2)

        while len(schedule_items) < max_stops:
            # 1) Find nearby from CURRENT position
            nearby = self.find_nearby_places(
                center_lat=current_lat,
                center_lng=current_lng,
                radius_km=radius_km,
                limit=20,
            )

            if include_debug_nearby:
                debug_nearby_snapshots.append(
                    {
                        "from": {"lat": current_lat, "lng": current_lng},
                        "nearby": nearby,
                    }
                )

            # 2) Build candidate list:
            #    - First: still-unvisited preferred (even if far, we allow it by not forcing nearby-only)
            #    - Then: nearby options
            candidates = []

            # preferred candidates (not yet visited)
            for pt in preferred_queue:
                if pt not in visited:
                    d = _dist_to(pt, current_lat, current_lng)
                    if d is not None:
                        candidates.append({"title": pt, "distance_km": d, "source": "preferred"})

            # nearby candidates (not yet visited)
            for p in nearby:
                title = p.get("title")
                if not title or title in visited:
                    continue
                candidates.append({"title": title, "distance_km": p.get("distance_km"), "source": "nearby"})

            # No candidates left
            if not candidates:
                break

            # 3) Choose the "best next" candidate
            #    Scoring approach (simple & effective):
            #    - lower crowd is better
            #    - closer distance is better
            #    - use best_visit_time() score already combines crowd + weather
            best_pick = None
            best_pick_score = -999999

            for c in candidates:
                title = c["title"]
                dist_km = c["distance_km"]
                if dist_km is None:
                    continue

                best_time = self.best_visit_time(title, base_day)
                if not best_time:
                    continue

                crowd = int(best_time["crowd_level"])
                visit_score = int(best_time["score"])  # 0-100, higher better

                # distance penalty: farther places slightly reduced
                # tweak this multiplier as you like
                route_score = visit_score - int(dist_km * 1.5)

                # if it's preferred, give it a small boost so it is not ignored
                if c["source"] == "preferred":
                    route_score += 10

                if route_score > best_pick_score:
                    best_pick_score = route_score
                    best_pick = {
                        "title": title,
                        "distance_km": dist_km,
                        "best_time": best_time,
                        "crowd": crowd,
                        "visit_score": visit_score,
                        "route_score": route_score,
                    }

            if not best_pick:
                break

            # 4) Allocate time (avoid duplicate same-hour collisions)
            visit_dt = datetime.strptime(best_pick["best_time"]["time"], "%Y-%m-%d %I:%M %p")
            used_hours = {datetime.strptime(i["time"], "%I:%M %p").hour for i in schedule_items if i.get("time")}
            while visit_dt.hour in used_hours and visit_dt.hour < 20:
                visit_dt += timedelta(hours=1)

            # 5) Add to schedule
            schedule_items.append(
                {
                    "order": len(schedule_items) + 1,
                    "time": visit_dt.strftime("%I:%M %p"),
                    "place": best_pick["title"],
                    "distance_from_prev_km": best_pick["distance_km"],
                    "score": best_pick["visit_score"],
                    "crowd": best_pick["crowd"],
                    "note": "High crowd" if best_pick["crowd"] > 60 else "Good time",
                }
            )

            visited.add(best_pick["title"])

            # 6) Update CURRENT center to this visited place (dynamic update you asked for)
            coords = self._get_place_coords(best_pick["title"])
            if coords:
                current_lat, current_lng = coords[0], coords[1]
            else:
                # if no coords, we can't update center — stop to avoid wrong distances
                break

        result = {"schedule": schedule_items}
        if include_debug_nearby:
            result["debug_nearby"] = debug_nearby_snapshots
        return result



    def _get_cached_weather_for_hour(self, dt: datetime) -> Dict[str, Any]:
        key = (dt.date().isoformat(), dt.hour)
        if key in self._weather_cache:
            return self._weather_cache[key]

        weather = self.get_weather_forecast(days=2)
        temp = None
        rain = 0.0
        for f in weather.get("forecast", []) or []:
            fdt = f.get("datetime")
            if isinstance(fdt, datetime) and fdt.date() == dt.date() and fdt.hour == dt.hour:
                temp = f.get("temperature")
                rain = f.get("rain", 0.0) or 0.0
                break

        rain_flag = 1 if rain and float(rain) > 0 else 0
        self._weather_cache[key] = {"temperature": temp, "rain_flag": rain_flag, "rain": rain}
        return self._weather_cache[key]


    def recommend_visit_times(
        self,
        place_name: str,
        date: datetime,
        start_hour: int = 6,
        end_hour: int = 21,
        step_minutes: int = 60,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Scan a full day for this place and return the best time slots
        based on crowd prediction + simple weather comfort scoring.

        Returns a list of dicts like:
        {
            "place": "<title>",
            "time": "YYYY-MM-DD HH:MM AM/PM",
            "score": 0–100 (higher = better),
            "crowd_level": 0–100,
            "label": "Low / Moderate / High crowd",
            "reasons": [ ... ]
        }
        """
        # Normalize place name/id using your existing map
        # (supports both "religious_1" and "Hanuman Vatika")
        if place_name in self.places_data:
            place = self.places_data[place_name]
            title = place.get("title", place_name)
        else:
            # maybe user passed id → try mapping
            norm = self.place_id_map.get(place_name)
            if norm and norm in self.places_data:
                place = self.places_data[norm]
                title = place.get("title", norm)
            else:
                # fall back to original behaviour (will raise from predict_crowd_level)
                title = place_name

        # --- get forecast for weather-aware scoring ---
        weather = self.get_weather_forecast(days=2)
        forecast = weather.get("forecast", []) or []

        # index weather by (date, hour)
        weather_by_hour: Dict[tuple, Dict[str, Any]] = {}
        for f in forecast:
            dt = f.get("datetime")
            if isinstance(dt, datetime):
                weather_by_hour[(dt.date(), dt.hour)] = f

        # time sweep
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        results: List[Dict[str, Any]] = []

        cur = day_start.replace(hour=start_hour, minute=0)
        end_dt = day_start.replace(hour=end_hour, minute=0)
        w = self._get_cached_weather_for_hour(cur)

        while cur <= end_dt:
            try:
                # 1) crowd prediction for this specific slot
                crowd_info = self.predict_crowd_level(title, cur)
                crowd = int(crowd_info["crowd_level"])
            except Exception:
                cur += timedelta(minutes=step_minutes)
                continue

            # 2) weather info for this hour (if available)
            wf = weather_by_hour.get((cur.date(), cur.hour), {})
            temp = wf.get("temperature")
            rain = wf.get("rain", 0.0) or 0.0

            # 3) scoring logic
            # Base: low crowd = good
            crowd_score = max(0, 100 - crowd)

            # temperature comfort (ideal around 25–30°C)
            temp_penalty = 0
            if temp is not None:
                # allow ~22–32°C comfortable; outside this gives penalty
                delta = max(0.0, abs(float(temp) - 27.0) - 5.0)
                temp_penalty = int(delta * 2)  # 2 points penalty per degree outside comfort band

            # rain penalty
            rain_penalty = int(min(30.0, float(rain) * 5.0))  # up to -30

            raw_score = crowd_score - temp_penalty - rain_penalty
            score = max(0, min(100, int(round(raw_score))))

            # crowd label
            if crowd <= 30:
                label = "Low crowd"
            elif crowd <= 60:
                label = "Moderate crowd"
            else:
                label = "High crowd"

            reasons = []
            reasons.append(f"Predicted crowd ~{crowd}%")
            if temp is not None:
                reasons.append(f"Temperature around {temp:.1f}°C")
            if rain > 0:
                reasons.append(f"Rain chance (≈{rain:.1f}mm)")

            results.append(
                {
                    "place": title,
                    "time": cur.strftime("%Y-%m-%d %I:%M %p"),
                    "score": score,
                    "crowd_level": crowd,
                    "label": label,
                    "reasons": reasons,
                }
            )

            cur += timedelta(minutes=step_minutes)

        # sort best → worst
        results.sort(key=lambda x: x["score"], reverse=True)

        # keep only top_k
        return results[:top_k]

    def best_visit_time(
        self,
        place_name: str,
        date: datetime,
        start_hour: int = 6,
        end_hour: int = 21,
        step_minutes: int = 60,
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience wrapper: just return the single best slot dict
        for this place on this date (or None if nothing could be computed).
        """
        slots = self.recommend_visit_times(
            place_name=place_name,
            date=date,
            start_hour=start_hour,
            end_hour=end_hour,
            step_minutes=step_minutes,
            top_k=1,
        )
        return slots[0] if slots else None

    def _get_cached_weather_for_hour(self, dt: datetime) -> Dict[str, Any]:
        key = (dt.date().isoformat(), dt.hour)
        if key in self._weather_cache:
            return self._weather_cache[key]

        weather = self.get_weather_forecast(days=2)
        temp = None
        rain = 0.0
        for f in weather.get("forecast", []) or []:
            fdt = f.get("datetime")
            if isinstance(fdt, datetime) and fdt.date() == dt.date() and fdt.hour == dt.hour:
                temp = f.get("temperature")
                rain = f.get("rain", 0.0) or 0.0
                break

        rain_flag = 1 if rain and float(rain) > 0 else 0
        self._weather_cache[key] = {"temperature": temp, "rain_flag": rain_flag, "rain": rain}
        return self._weather_cache[key]


    def _create_weather_context(self, daily_weather: Dict) -> str:
        if not daily_weather:
            return "No reliable forecast; plan for flexible indoor/outdoor options."
        lines = []
        for day, weather in daily_weather.items():
            date_str = weather["date"].strftime("%A, %B %d")
            wet = "Rainy" if weather["rain"] > 2 else "Dry"
            lines.append(
                f"Day {day + 1} ({date_str}): "
                f"{weather['avg_temperature']:.1f}°C, {wet}, {weather['description']}"
            )
        return "\n".join(lines)

    def _enhance_schedule_with_predictions(
        self, schedule_text: str, daily_weather: Dict
    ) -> str:
        """Enhance schedule with specific crowd predictions"""
        enhanced_schedule = schedule_text

        weather_summary = "\n=== WEATHER & CROWD INTELLIGENCE ===\n"
        for day, weather in daily_weather.items():
            date_str = weather["date"].strftime("%A, %B %d")
            weather_summary += f"Day {day + 1} ({date_str}): "
            weather_summary += f"{weather['avg_temperature']:.1f}°C, "

            if weather["rain"] > 5:
                weather_summary += (
                    "Heavy rain expected - Focus on indoor activities\n"
                )
            elif weather["rain"] > 2:
                weather_summary += (
                    "Light rain possible - Carry umbrella, indoor backup plans ready\n"
                )
            elif weather["avg_temperature"] > 35:
                weather_summary += (
                    "Hot weather - Early morning and evening visits recommended\n"
                )
            else:
                weather_summary += (
                    "Pleasant weather - Great for all outdoor activities\n"
                )

        weather_summary += "\n=== ORIGINAL SCHEDULE ===\n"
        enhanced_schedule = weather_summary + enhanced_schedule

        return enhanced_schedule

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def _categorize_place(self, place_id: str) -> str:
        if place_id.startswith("religious"):
            return "Religious Site"
        elif place_id.startswith("nature"):
            return "Nature/Recreation"
        elif place_id.startswith("general"):
            return "General Information"
        else:
            return "Tourist Attraction"

    def _get_best_for_category(self, place_id: str) -> str:
        if place_id.startswith("religious"):
            return "Spiritual experience, photography, cultural immersion"
        elif place_id.startswith("nature"):
            return "Family outings, photography, relaxation, adventure"
        else:
            return "Sightseeing, cultural experience"

    def _guess_cuisine_type(self, name: str) -> str:
        name_lower = name.lower()
        if any(word in name_lower for word in ["pizza", "domino"]):
            return "Italian/Fast Food"
        elif any(word in name_lower for word in ["dosa", "idli", "south"]):
            return "South Indian"
        elif any(word in name_lower for word in ["sweet", "mithai"]):
            return "Sweets/Desserts"
        elif any(word in name_lower for word in ["fast food", "momos", "chaat"]):
            return "Fast Food/Street Food"
        elif "hotel" in name_lower:
            return "Multi-cuisine"
        else:
            return "Indian/Multi-cuisine"

    def get_recommendations(self, query: str, days: int = 1) -> Dict[str, Any]:
        """
        Provide recommendations by combining weather forecast,
        WeatherAwareRecommendationEngine, and QA chain.
        """
        weather = self.get_current_weather(api_key=self.weather_api_key, days=days)
        current = (
            weather["forecast"][0]
            if "forecast" in weather and weather["forecast"]
            else {}
        )

        condition = current.get("description", "clear")
        temp = current.get("temperature", 28)
        rain = current.get("rain", 0.0)

        rec_engine = WeatherAwareRecommendationEngine(self)
        weather_recs = rec_engine.get_weather_appropriate_activities(
            weather_condition=condition, temperature=temp, rain_mm=rain
        )

        query_lower = query.lower()
        filtered_recs = []
        if "indoor" in query_lower:
            filtered_recs = [
                r
                for r in weather_recs
                if "Indoor" in r["activity"] or "Temple" in r["activity"]
            ]
        elif "outdoor" in query_lower:
            filtered_recs = [
                r
                for r in weather_recs
                if "Outdoor" in r["activity"] or "Park" in r["activity"]
            ]

        if filtered_recs:
            return {
                "answer": (
                    "Here are indoor/outdoor activities in Rourkela based on your "
                    f"query and weather ({condition}, {temp}°C):"
                ),
                "weather_recommendations": filtered_recs,
                "sources": ["WeatherAwareRecommendationEngine"],
            }

        if weather_recs:
            qa_answer = None
            try:
                qa_answer = self.qa_chain.invoke(query)
            except Exception:
                qa_answer = None  # LLM down / key blocked etc.

            resp = {
                "answer": (
                    f"Based on the weather ({condition}, {temp}°C, rain={rain}mm), "
                    "here are suggestions:"
                ),
                "weather_recommendations": weather_recs,
                "sources": ["WeatherAwareRecommendationEngine"],
            }
            if qa_answer:
                resp["qa_answer"] = qa_answer
            return resp


        # fallback to QA chain only
        try:
            answer = self.qa_chain.invoke(query)
        except Exception as e:
            answer = f"(LLM unavailable) {e}"
        return {"answer": answer, "sources": []}


    def _estimate_price_range(self, name: str, rating: float) -> str:
        name_lower = name.lower()
        if any(
            word in name_lower
            for word in ["hotel", "palace", "regency", "international"]
        ):
            return "expensive" if rating > 4.0 else "moderate"
        elif any(
            word in name_lower for word in ["fast food", "stall", "chaat", "street"]
        ):
            return "budget"
        elif rating and rating > 4.2:
            return "moderate"
        else:
            return "budget"

    def _get_restaurant_suitable_for(self, name: str, rating: float) -> str:
        name_lower = name.lower()
        if "hotel" in name_lower and rating > 4.0:
            return "Family dining, business meals, special occasions"
        elif any(word in name_lower for word in ["fast food", "pizza"]):
            return "Quick meals, casual dining, groups"
        elif any(word in name_lower for word in ["sweet", "snacks"]):
            return "Desserts, tea time, gifts"
        else:
            return "Casual dining, local experience"

def main():
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    WEATHER_API_KEY = os.getenv("WHEATHER_KEY") 
    planner = RourkelalTourismSchedulePlanner(
        api_key=API_KEY,
        weather_api_key=WEATHER_API_KEY,
        attractions_file="rag.json",
        restaurants_file="restaurants_rourkela.json",
        model_file="rkl_places_xgb.pkl",
    )

    now = datetime.now()
    day_date = datetime(2026, 1, 4)
    start_lat = 22.251251315985133
    start_lng = 84.90486268773235

    dynamic = planner.build_day_schedule_dynamic_route(
        date=day_date,
        start_lat=start_lat,
        start_lng=start_lng,
        preferred_titles=["Hanuman Vatika", "Vedvyas Temple"],
        radius_km=60.0,
        max_stops=4,
        include_debug_nearby=False
    )

    print("\n=== DYNAMIC ROUTE SCHEDULE JSON ===")
    print(json.dumps({
        "date": day_date.strftime("%Y-%m-%d"),
        "center": {"lat": start_lat, "lng": start_lng},
        "schedule": dynamic["schedule"]
    }, indent=2, ensure_ascii=False))

    print("=== Smart 3-Day Schedule with Weather & Crowd Intelligence ===")
    smart_schedule = planner.create_smart_schedule(
        duration_days=3,
        interests=["religious", "nature", "food"],
        budget="moderate",
        group_type="family",
        start_date=now,
    )

    print(smart_schedule["schedule"])
    print("\nWeather Summary:")
    for day, weather in smart_schedule["weather_forecast"].items():
        print(
            f"Day {day+1}: {weather['avg_temperature']:.1f}°C, "
            f"Rain: {weather['rain']:.1f}mm"
        )

    print("\n=== Crowd Prediction Example ===")
    visit_time = now
    crowd_info = planner.predict_crowd_level("Hanuman Vatika", visit_time)
    print(f"Hanuman Vatika at {visit_time.strftime('%A %I:%M %p')}:")
    print(
            f"Crowd Level: {crowd_info['crowd_level']}% - "
            f"{crowd_info['description']}"
    )
    print(f"Better times: {crowd_info['best_alternative_times']}")

    print("\n=== Weather-Based Recommendations ===")
    weather_query = planner.get_recommendations(
        "What are the best indoor activities in Rourkela for a rainy day with family?"
    )
    print(weather_query["answer"])
    if "weather_recommendations" in weather_query:
        for rec in weather_query["weather_recommendations"]:
            print(
                f"- {rec['activity']}: "
                f"{', '.join(rec['places'])} ({rec['reason']})"
            )

    print("\n=== 5-Day Weather Forecast ===")
    weather = planner.get_weather_forecast(5)
    print(f"Weather source: {weather['source']}")
    for forecast in weather["forecast"][:8]:
        print(
            f"{forecast['datetime'].strftime('%Y-%m-%d %H:%M')}: "
            f"{forecast['temperature']:.1f}°C, {forecast['description']}, "
            f"Rain: {forecast['rain']:.1f}mm"
        )


def create_crowd_data_file():
    """Helper function to create a sample crowd prediction file"""
    sample_crowd_data = {
        "religious_1": {
            "place_name": "Hanuman Vatika",
            "daily_base_crowds": {
                "monday": 40,
                "tuesday": 65,
                "wednesday": 35,
                "thursday": 40,
                "friday": 55,
                "saturday": 85,
                "sunday": 90,
            },
            "hourly_patterns": {
                "early_morning": {"multiplier": 0.4, "hours": [6, 7, 8]},
                "morning": {"multiplier": 0.7, "hours": [9, 10, 11]},
                "afternoon": {
                    "multiplier": 0.6,
                    "hours": [12, 13, 14, 15],
                },
                "evening": {
                    "multiplier": 1.0,
                    "hours": [16, 17, 18, 19],
                },
                "night": {"multiplier": 0.5, "hours": [20, 21, 22]},
            },
            "festival_multipliers": {
                "hanuman_jayanti": 3.5,
                "tuesday_special": 1.8,
                "diwali": 2.5,
            },
            "weather_sensitivity": {
                "rain": 0.8,
                "hot_weather": 0.7,
                "cold_weather": 0.9,
            },
            "capacity_level": "high",
        },
        "nature_1": {
            "place_name": "Indira Gandhi Park",
            "daily_base_crowds": {
                "monday": 25,
                "tuesday": 20,
                "wednesday": 25,
                "thursday": 30,
                "friday": 40,
                "saturday": 80,
                "sunday": 85,
            },
            "hourly_patterns": {
                "early_morning": {"multiplier": 0.3, "hours": [6, 7, 8]},
                "morning": {"multiplier": 0.6, "hours": [9, 10, 11]},
                "afternoon": {
                    "multiplier": 0.4,
                    "hours": [12, 13, 14, 15],
                },
                "evening": {
                    "multiplier": 1.0,
                    "hours": [16, 17, 18, 19],
                },
                "night": {"multiplier": 0.2, "hours": [20, 21, 22]},
            },
            "festival_multipliers": {
                "children_day": 2.0,
                "republic_day": 1.5,
                "independence_day": 1.8,
            },
            "weather_sensitivity": {
                "rain": 0.2,
                "hot_weather": 0.5,
                "cold_weather": 0.8,
            },
            "capacity_level": "high",
        },
    }

    with open("crowd_predictions.json", "w", encoding="utf-8") as f:
        json.dump(sample_crowd_data, f, indent=2, ensure_ascii=False)

    print("Created sample crowd_predictions.json file")


class WeatherAwareRecommendationEngine:
    """Additional class for weather-specific recommendations"""

    def __init__(self, planner: RourkelalTourismSchedulePlanner):
        self.planner = planner

    def get_weather_appropriate_activities(
        self, weather_condition: str, temperature: float, rain_mm: float = 0
    ) -> List[Dict]:
        """Get activities suitable for specific weather conditions"""
        recommendations = []

        # Rainy weather recommendations
        if rain_mm > 2 or any(
            word in weather_condition.lower()
            for word in ["rain", "thunderstorm", "drizzle"]
        ):
            recommendations.extend(
                [
                    {
                        "activity": "Temple visits",
                        "places": [
                            "Hanuman Vatika",
                            "Vedvyas Temple",
                            "Jagannath Temple",
                        ],
                        "reason": "Covered areas, spiritual experience unaffected by rain",
                        "timing": "Any time, avoid travel during heavy downpours",
                    },
                    {
                        "activity": "Indoor dining",
                        "places": [
                            "Hotel Radhika Regency",
                            "Curry Pot",
                            "Pizza Den",
                        ],
                        "reason": "Comfortable indoor seating, good ambiance",
                        "timing": "Extended meal times, perfect for long conversations",
                    },
                    {
                        "activity": "Shopping and local markets",
                        "places": [
                            "Covered markets in sectors",
                            "Shopping complexes",
                        ],
                        "reason": "Browse local handicrafts and souvenirs",
                        "timing": "Afternoon hours when rain is typically lighter",
                    },
                ]
            )

        # Hot weather recommendations (>35°C)
        elif temperature > 35:
            recommendations.extend(
                [
                    {
                        "activity": "Early morning temple visits",
                        "places": ["Hanuman Vatika", "Vaishno Devi Temple"],
                        "reason": "Peaceful atmosphere, cooler temperatures",
                        "timing": "6:00-9:00 AM",
                    },
                    {
                        "activity": "Air-conditioned restaurants",
                        "places": [
                            "Hotel Radhika Regency",
                            "The Sarovar Court",
                        ],
                        "reason": "Comfortable dining environment",
                        "timing": "12:00-3:00 PM (lunch break from heat)",
                    },
                    {
                        "activity": "Evening park visits",
                        "places": ["Indira Gandhi Park", "Green Park"],
                        "reason": "Cooler evening breeze, family activities",
                        "timing": "5:00-7:00 PM",
                    },
                ]
            )

        # Pleasant weather recommendations (15-30°C)
        else:
            recommendations.extend(
                [
                    {
                        "activity": "Outdoor sightseeing",
                        "places": [
                            "Mandira Dam",
                            "Pitamahal Dam",
                            "Darjeeng Picnic Spot",
                        ],
                        "reason": "Perfect weather for photography and nature walks",
                        "timing": "Any time, especially 9:00 AM - 5:00 PM",
                    },
                    {
                        "activity": "Trekking and adventure",
                        "places": [
                            "Kanha Kund",
                            "Ghagara Waterfall",
                            "Ghogar Natural Site",
                        ],
                        "reason": "Ideal conditions for outdoor activities",
                        "timing": "Early morning or late afternoon for best experience",
                    },
                    {
                        "activity": "Riverside relaxation",
                        "places": [
                            "Koel Riverbank",
                            "Vedvyas Temple confluence",
                        ],
                        "reason": "Serene environment, perfect for meditation",
                        "timing": "Sunset hours for beautiful views",
                    },
                ]
            )

        return recommendations

    def create_weather_adaptive_itinerary(
        self, base_itinerary: List[Dict], weather_forecast: List[Dict]
    ) -> List[Dict]:
        """Adapt an existing itinerary based on weather forecast"""
        adapted_itinerary = []

        for day_plan in base_itinerary:
            day_weather = weather_forecast[day_plan.get("day", 0)]

            adaptations = {
                "original_plan": day_plan,
                "weather_context": day_weather,
                "adaptations": [],
                "backup_plans": [],
            }

            if day_weather.get("rain", 0) > 5:
                adaptations["adaptations"].append(
                    "Heavy rain expected - Move outdoor activities to covered areas"
                )
                adaptations["backup_plans"].extend(
                    [
                        "Visit temples with covered walkways",
                        "Extended dining experiences at indoor restaurants",
                        "Local shopping in covered markets",
                    ]
                )

            elif day_weather.get("temperature", 25) > 35:
                adaptations["adaptations"].append(
                    "Hot weather - Shift schedule to early morning and evening"
                )
                adaptations["backup_plans"].extend(
                    [
                        "6:00-9:00 AM: Temple visits and morning activities",
                        "12:00-4:00 PM: Indoor dining and rest",
                        "5:00-8:00 PM: Parks and outdoor sightseeing",
                    ]
                )

            adapted_itinerary.append(adaptations)

        return adapted_itinerary


# Additional utility functions
def analyze_crowd_patterns(crowd_data: Dict) -> Dict[str, Any]:
    """Analyze crowd patterns to provide insights"""
    analysis = {
        "peak_days": [],
        "best_days": [],
        "peak_hours": [],
        "best_hours": [],
        "seasonal_trends": {},
    }

    all_daily_crowds: Dict[str, List[float]] = {}
    for place_data in crowd_data.values():
        daily_crowds = place_data.get("daily_base_crowds", {})
        for day, crowd in daily_crowds.items():
            all_daily_crowds.setdefault(day, []).append(crowd)

    avg_daily_crowds = {
        day: sum(crowds) / len(crowds) for day, crowds in all_daily_crowds.items()
    }

    sorted_days = sorted(avg_daily_crowds.items(), key=lambda x: x[1])
    analysis["best_days"] = [day for day, _ in sorted_days[:2]]
    analysis["peak_days"] = [day for day, _ in sorted_days[-2:]]

    peak_hours = set()
    best_hours = set()

    for place_data in crowd_data.values():
        hourly_patterns = place_data.get("hourly_patterns", {})
        for pattern in hourly_patterns.values():
            mult = pattern.get("multiplier", 1.0)
            hours = pattern.get("hours", [])
            if mult > 0.8:
                peak_hours.update(hours)
            elif mult < 0.5:
                best_hours.update(hours)

    analysis["peak_hours"] = sorted(list(peak_hours))
    analysis["best_hours"] = sorted(list(best_hours))

    return analysis


if __name__ == "__main__":
    if not os.path.exists("crowd_predictions.json"):
        create_crowd_data_file()
    main()
