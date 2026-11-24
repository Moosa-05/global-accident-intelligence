"""Pydantic schemas for Human Injury Severity Prediction (3-Class).
Predicts collision-level physiological trauma outcome:
- SLIGHT
- SERIOUS
- FATAL
"""
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid
from backend.app.core.units import UnitSystem
from backend.app.schemas.severity import GeographicMetadata

class InjurySeverityPredictionRequest(BaseModel):
    # Location (WGS84 EPSG:4326)
    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 51.5074}, description="WGS84 Latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": -0.1278}, description="WGS84 Longitude")
    
    # Road characteristics
    road_class: str = Field("PRIMARY_ARTERIAL", description="Canonical road class: MOTORWAY, TRUNK_HIGHWAY, PRIMARY_ARTERIAL, SECONDARY_COLLECTOR, LOCAL_RESIDENTIAL, LOCAL_RURAL")
    speed_limit_kmh: Optional[float] = Field(50.0, ge=10.0, le=200.0, description="Speed limit in km/h")
    divided_roadway: bool = Field(False, description="Physically divided dual carriageway or motorway")
    is_junction: bool = Field(False, description="Incident on or approaching junction/intersection")
    is_traffic_signal: bool = Field(False, description="Traffic signal controlled")
    is_crossing: bool = Field(False, description="Pedestrian crossing proximity")
    is_roundabout: bool = Field(False, description="Roundabout / rotary junction")
    is_stop_sign: bool = Field(False, description="Stop sign controlled")
    is_give_way: bool = Field(False, description="Yield / Give way controlled")
    is_railway: bool = Field(False, description="Railway level crossing")
    
    # Temporal & Astronomical
    timestamp: Optional[datetime] = Field(None, description="Event timestamp (defaults to UTC now)")
    timezone: Optional[str] = Field("UTC", description="IANA Time Zone identifier")
    is_night: Optional[bool] = Field(False, description="Darkness / nighttime indicator")
    
    # Environmental & Atmospheric (Metric standard)
    units: UnitSystem = Field(UnitSystem.METRIC, description="Input unit system: 'metric' or 'imperial'")
    temperature: Optional[float] = Field(15.0, description="Ambient air temperature")
    humidity: Optional[float] = Field(70.0, ge=0.0, le=100.0, description="Relative humidity %")
    pressure: Optional[float] = Field(1013.25, description="Surface pressure in hPa or inHg")
    wind_speed: Optional[float] = Field(15.0, ge=0.0, description="Wind speed in km/h or mph")
    precipitation: Optional[float] = Field(0.0, ge=0.0, description="Precipitation rate in mm or inches")
    weather_condition: Optional[str] = Field("CLEAR", description="Weather category: CLEAR, CLOUDY_OVERCAST, LIGHT_RAIN, HEAVY_RAIN, SNOW_HAIL, FOG_SMOKE, HIGH_WINDS_STORM")
    light_condition: Optional[str] = Field("DAYLIGHT", description="Light condition: DAYLIGHT, TWILIGHT_DAWN_DUSK, DARKNESS_STREETLIGHTS_ON, DARKNESS_NO_STREETLIGHTS")
    road_surface_condition: Optional[str] = Field("DRY", description="Surface condition: DRY, WET_DAMP, SNOW_SLUSH, FROST_ICE, FLOOD")
    
    # Client alias fields for backwards compatibility and frontend simulations
    light_conditions: Optional[str] = Field(None, description="Client alias for light_condition")
    weather_conditions: Optional[str] = Field(None, description="Client alias for weather_condition")
    surface_conditions: Optional[str] = Field(None, description="Client alias for road_surface_condition")
    vehicle_type: Optional[str] = Field(None, description="Primary vehicle type context")
    number_of_vehicles: Optional[int] = Field(None, description="Vehicle count involved")
    number_of_casualties: Optional[int] = Field(None, description="Casualty count involved")
    urban_or_rural: Optional[str] = Field(None, description="Urban or rural environment setting")

class InjurySeverityPredictionResponse(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    predicted_severity: str = Field(..., description="'SLIGHT', 'SERIOUS', or 'FATAL'")
    severity_level_code: int = Field(..., description="0: SLIGHT, 1: SERIOUS, 2: FATAL")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities for SLIGHT, SERIOUS, FATAL (sums to 1.0)")
    fatality_risk_alert: bool = Field(..., description="True if fatal casualty probability exceeds elevated risk threshold (>10%)")
    target_definition: str = "Human Physiological Injury Severity (Conditional on Collision: UNECE / WHO 30-day definition; conditional on collision occurrence)"
    model_version: str = "1.0.0"
    training_data_sources: str = "UK STATS19 (OGL v3.0) & France BAAC (Licence Ouverte v2.0)"
    latency_ms: float
    timestamp: str
    geographic_metadata: Optional[GeographicMetadata] = None
