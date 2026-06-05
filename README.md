# Rourkela Tourism Planner 🗺️

An AI-powered **day & multi-day trip planner** for Rourkela (Odisha, India). Given your
location (lat/lng) and preferences, it builds an optimized itinerary that accounts for
**predicted crowd levels**, **live weather**, and **travel time between stops** — and can
generate natural-language itineraries using a Gemini-backed RAG pipeline grounded in a
curated dataset of local attractions and restaurants.

> Built for HACKNITR.

---

## ✨ Features

- **Crowd prediction** — an XGBoost model ([rkl_places_xgb.pkl](rkl_places_xgb.pkl)) predicts
  the crowd level (%) for any place at any hour, using engineered features (category, weekday,
  month, hour curves, holidays/festivals, season, and live weather).
- **Weather-aware planning** — pulls live data from [WeatherAPI](https://www.weatherapi.com/)
  and falls back to seasonal sample data when no key is set. Avoids hot midday slots and rainy
  outdoor visits.
- **Lat/Lng day planner** — finds nearby attractions within a radius, scores time slots, and
  picks non-overlapping stops with realistic travel buffers.
- **Dynamic routing** — recomputes the "next best place" after each stop (greedy nearest-friendly
  routing combining crowd + distance scores).
- **RAG itineraries** — retrieval-augmented generation over local data (FAISS locally, or a
  Pathway vector-search microservice) + `gemini-2.5-flash`, with an allowlist so the LLM only
  recommends real places.
- **REST API** — a FastAPI service exposing `/plan-day`.
- **Twitter scraper** — optional utility to pull recent tweets for a hashtag/location.

---

## 🧱 Architecture

```
                         ┌─────────────────────────────┐
   POST /plan-day  ───▶  │  api (flask_test.py, FastAPI)│
                         └──────────────┬──────────────┘
                                        │
                    ┌───────────────────▼─────────────────────┐
                    │ LatLngSchedulePlanner (place_recommender)│  ← nearby + time-slot selection
                    └───────────────────┬─────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │ RourkelalTourismSchedulePlanner (rag.py)            │
              │  • crowd model (XGBoost)                            │
              │  • weather (WeatherAPI / sample fallback)           │
              │  • RAG QA chain (Gemini + retrieval)                │
              └───────┬───────────────────────────┬───────────────-┘
                      │                            │
        ┌─────────────▼───────────┐    ┌───────────▼────────────────────┐
        │ data_converter.py       │    │ Retrieval                       │
        │ feature engineering for │    │  • FAISS (faiss_index/) local   │
        │ the crowd model         │    │  • Pathway service :8765 (RAG)  │
        └─────────────────────────┘    └─────────────────────────────────┘
```

### Key files

| File | Purpose |
|------|---------|
| [rag.py](rag.py) | Core planner: data loading, crowd prediction, weather, RAG QA chain, smart/day schedules. |
| [place_recommender.py](place_recommender.py) | `LatLngSchedulePlanner` — lat/lng day planning with travel-time buffers. |
| [data_converter.py](data_converter.py) | `CrowdPredictionInputProcessor` — builds the 18-feature vector for the XGBoost model. |
| [flask_test.py](flask_test.py) | FastAPI app (`/health`, `/plan-day`). *(despite the name, this is the FastAPI service)* |
| [pathway_service.py](pathway_service.py) | Standalone vector-search microservice (`/search`, `/health`) on port `8765`. |
| [pathway_retriever_client.py](pathway_retriever_client.py) | HTTP client used by the RAG chain to query the Pathway service. |
| [twitter_bot.py](twitter_bot.py) | CLI to extract recent tweets by hashtag + location. |
| [test.py](test.py) / [input_pipeline.py](input_pipeline.py) | Runnable examples for the planner and the crowd model. |
| [rag.json](rag.json) | Attractions dataset (id, title, lat/lng, description). |
| [restaurants_rourkela.json](restaurants_rourkela.json) | Restaurants dataset (name, address, rating, lat/lng). |
| [rkl_places_xgb.pkl](rkl_places_xgb.pkl) | Trained crowd-prediction model. |
| [faiss_index/](faiss_index/) | Prebuilt FAISS vector index (loaded if present, else rebuilt). |

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (required for RAG itineraries)
- *(Optional)* A [WeatherAPI](https://www.weatherapi.com/) key for live weather (sample data is used otherwise)

### 2. Clone & set up a virtual environment

```powershell
cd python_files
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux
```

### 3. Install dependencies

```powershell
pip install -r Requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in `python_files/` (it is git-ignored):

```dotenv
GEMINI_API_KEY=your_gemini_key_here
WHEATHER_KEY=your_weatherapi_key_here   # note: env var is spelled "WHEATHER_KEY"
# TWITTER_BEARER_KEY=your_twitter_bearer_token   # only for twitter_bot.py
```

> ⚠️ The env variable for the weather key is intentionally spelled `WHEATHER_KEY` to match the
> code. Without it, the planner falls back to synthetic seasonal weather data and still works.

### 5. Run the planner (no server needed)

The fastest way to see it work end-to-end:

```powershell
python test.py
```

This plans a day around central Rourkela with two preferred places and prints the schedule,
weather summary, and nearby places. You can also run the richer demo in [rag.py](rag.py):

```powershell
python rag.py
```

On first run this builds the FAISS index from the JSON datasets (cached in `faiss_index/`).

---

## 🌐 Running the REST API

The FastAPI app lives in [flask_test.py](flask_test.py) (the `app` object).

```powershell
uvicorn flask_test:app --reload --port 8000
```

Then open the interactive docs at **http://localhost:8000/docs**.

### Example request

```bash
curl -X POST http://localhost:8000/plan-day \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 22.2396,
    "lng": 84.8633,
    "max_stops": 4,
    "start_hour": 8,
    "end_hour": 20,
    "preferred_places": ["Hanuman Vatika", "Indira Gandhi Park"]
  }'
```

### `POST /plan-day` — request fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lat`, `lng` | float | — | **Required.** User location. |
| `date` | ISO datetime | now | Target date. |
| `radius_km` | float | server default (6 km) | Search radius for nearby places. |
| `max_stops` | int | 6 | Max places in the itinerary. |
| `start_hour` / `end_hour` | int (0–23) | 8 / 20 | Day window. |
| `preferred_places` | string[] | — | Place titles (must match `rag.json`). If given, crowd modeling is enabled and these places drive the plan. |
| `include_nearby` | bool | true | Include the raw nearby-places list in the response. |

The response contains `date`, `center`, `weather_summary`, an ordered `schedule` (time, place,
score, crowd %, travel time), and optionally `nearby_places`.

`GET /health` returns a simple readiness check.

---

## 🔍 Optional: Pathway vector-search service (full RAG)

The RAG QA chain in [rag.py](rag.py) retrieves context from a **Pathway** microservice on
`http://127.0.0.1:8765`. The day-planning endpoints work without it, but LLM-generated
narrative itineraries (`create_smart_schedule`, `get_recommendations`) need it running.

Start it with Docker Compose:

```powershell
docker compose up --build
```

This launches [pathway_service.py](pathway_service.py), which embeds the attraction/restaurant
datasets with `sentence-transformers/all-MiniLM-L6-v2` and serves a `/search` endpoint.

Verify: `curl http://localhost:8765/health`

---

## 🐦 Optional: Twitter scraper

```powershell
python twitter_bot.py --hashtag Rourkela --location Rourkela --max 50 --output tweets.csv
```

Requires `TWITTER_BEARER_KEY` in your `.env`.

---

## 🛠️ Tech stack

- **LLM / RAG:** LangChain, `langchain-google-genai` (Gemini 2.5 Flash), FAISS, Pathway,
  sentence-transformers
- **ML:** XGBoost, scikit-learn, NumPy, pandas
- **API:** FastAPI, Uvicorn, Pydantic
- **Data:** WeatherAPI, Tweepy
- **Geo:** Haversine distance + greedy nearest-neighbor routing

---

## 📝 Notes & caveats

- The dataset and crowd model are **Rourkela-specific**; place names in `preferred_places` must
  match titles in [rag.json](rag.json).
- [data_converter.py](data_converter.py) currently contains a hardcoded WeatherAPI key — move it
  to the `.env` (`WHEATHER_KEY`) before deploying.
- Holiday/festival multipliers are defined through 2025; extend `holiday_rules` in
  [data_converter.py](data_converter.py) for future dates.
- If the FAISS index gets stale after editing the datasets, delete the `faiss_index/` folder to
  force a rebuild on the next run.
