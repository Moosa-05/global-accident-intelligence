from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid
from backend.app.core.units import UnitSystem

class GeographicMetadata(BaseModel):
    coordinates: Dict[str, float]
    timezone: Optional[str] = "UTC"
    is_out_of_distribution: bool = Field(..., description="True if coordinates are outside empirical training geography")
    validation_status: str = Field(..., description="'EMPIRICALLY_VALIDATED' or 'UNVALIDATED_TERRITORY'")
    validation_warning: Optional[str] = Field(None, description="Clear notice of extrapolation when evaluated outside training geography")
    training_geography: str
    licensing_terms: str

class SeverityPredictionRequest(BaseModel):
    # Global Coordinates (accepts all valid WGS84 coordinates)
    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 35.0456}, description="WGS84 Latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": -85.3097}, description="WGS84 Longitude")
    
    # Temporal & Timezone
    timestamp: Optional[datetime] = Field(None, description="Observation timestamp (defaults to UTC now)")
    timezone: Optional[str] = Field("UTC", description="IANA Time Zone identifier (e.g. 'America/New_York', 'Europe/London')")
    
    # Measurement System
    units: UnitSystem = Field(UnitSystem.METRIC, description="Input unit system: 'metric' (C, km/h, mm, hPa) or 'imperial' (F, mph, in, inHg)")
    
    # Atmospheric / Weather Context
    temperature: Optional[float] = Field(20.0, description="Ambient temperature (°C or °F depending on units)")
    humidity: Optional[float] = Field(60.0, ge=0.0, le=100.0, description="Relative humidity percentage")
    pressure: Optional[float] = Field(1013.25, description="Barometric pressure (hPa or inHg)")
    visibility: Optional[float] = Field(10.0, ge=0.0, description="Atmospheric visibility (km or miles)")
    wind_speed: Optional[float] = Field(10.0, ge=0.0, description="Wind speed (km/h or mph)")
    precipitation: Optional[float] = Field(0.0, ge=0.0, description="Precipitation rate (mm or inches)")
    weather_condition: Optional[str] = Field("Clear", description="Weather text (Clear, Rain, Snow, Fog, etc.)")
    
    # Global Road Infrastructure Attributes (binary features independent of country)
    is_junction: bool = Field(False, description="Located on or near a highway/roadway junction")
    is_crossing: bool = Field(False, description="Pedestrian or street crossing")
    is_traffic_signal: bool = Field(False, description="Traffic signal controlled")
    is_stop_sign: bool = Field(False, description="Stop sign controlled")
    is_station: bool = Field(False, description="Transit/bus/rail station proximity")
    is_railway: bool = Field(False, description="Railway level crossing")
    is_roundabout: bool = Field(False, description="Traffic roundabout / rotary")
    is_give_way: bool = Field(False, description="Yield / Give Way controlled")
    is_traffic_calming: bool = Field(False, description="Speed bump / traffic calming")
    is_no_exit: bool = Field(False, description="Dead-end or no-exit street")
    is_amenity: bool = Field(False, description="Commercial amenity proximity")

class SeverityPredictionResponse(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    predicted_severity: int = Field(..., ge=1, le=4, description="Most likely traffic-impact severity (1: Minimal, 2: Moderate, 3: Significant, 4: Severe)")
    severity_level: str = Field(..., description="Semantic category: MINIMAL_IMPACT, MODERATE_IMPACT, SIGNIFICANT_IMPACT, SEVERE_DISRUPTION")
    severity_probabilities: Dict[str, float] = Field(..., description="Class probabilities for all 4 severity levels (sums to 1.0)")
    severity_description: str
    model_version: str
    latency_ms: float
    timestamp: str
    geographic_metadata: GeographicMetadata
