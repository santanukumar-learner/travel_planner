# api.py

import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag import RourkelalTourismSchedulePlanner
from place_recommender import LatLngSchedulePlanner


app = FastAPI(
    title="Rourkelal Tourism Planner API",
    description="Lat/Lng based day planner with crowd & weather intelligence",
    version="1.0.0",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WHEATHER_KEY")  

# Initialize core planner
planner = RourkelalTourismSchedulePlanner(
    api_key=GEMINI_API_KEY,
    weather_api_key=WEATHER_API_KEY,
    attractions_file="rag.json",
    restaurants_file="restaurants_rourkela.json",
    model_file="rkl_places_xgb.pkl",
)

# Lat/Lng schedule wrapper
geo = LatLngSchedulePlanner(
    planner,
    default_radius_km=6.0,
    dwell_minutes=60,
    travel_speed_kmh=20.0,
)



class PlanDayRequest(BaseModel):
    lat: float = Field(..., description="User latitude")
    lng: float = Field(..., description="User longitude")
    date: Optional[datetime] = Field(
        None,
        description="Target date (ISO format). If omitted, uses current date/time."
    )
    radius_km: Optional[float] = Field(
        None,
        description="Search radius in km. If omitted, uses server default."
    )
    max_stops: int = Field(
        6,
        description="Maximum number of places to include in the schedule."
    )
    start_hour: int = Field(
        8,
        ge=0, le=23,
        description="Day start hour in 24h format (local time)."
    )
    end_hour: int = Field(
        20,
        ge=0, le=23,
        description="Day end hour in 24h format (local time)."
    )
    preferred_places: Optional[List[str]] = Field(
        None,
        description="Optional list of place names (must match titles in rag.json). "
                    "If provided, planner will only consider these places."
    )
    include_nearby: bool = Field(
        True,
        description="Whether to include raw nearby place list in the response."
    )


class PlanDayResponse(BaseModel):
    date: str
    center: Dict[str, float]
    weather_summary: str
    schedule: List[Dict[str, Any]]
    nearby_places: Optional[List[Dict[str, Any]]] = None



@app.get("/health")
def health_check():
    """Simple health check."""
    return {"status": "ok", "planner_loaded": True}


@app.post("/plan-day", response_model=PlanDayResponse)
def plan_day(req: PlanDayRequest):
    """
    Plan a day in Rourkela based on user lat/lng and optional preferred places.
    """
    target_date = req.date or datetime.now()

    day_plan = geo.plan_day(
        lat=req.lat,
        lng=req.lng,
        date=target_date,
        radius_km=req.radius_km,
        max_stops=req.max_stops,
        start_hour=req.start_hour,
        end_hour=req.end_hour,
        preferred_places=req.preferred_places,
        include_nearby=req.include_nearby,
    )

    # FastAPI will auto-convert normal dict to PlanDayResponse
    return day_plan
