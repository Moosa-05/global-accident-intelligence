from typing import Optional, List
from pydantic import BaseModel, Field
import uuid

class ObservationItem(BaseModel):
    speed: float = Field(..., description="Target sensor speed (mph)")
    volume: float = Field(..., description="Target sensor traffic volume count")
    occupancy: float = Field(..., description="Target sensor lane occupancy (%)")
    speed_upstream: Optional[float] = None
    speed_downstream: Optional[float] = None

class RiskPredictionRequest(BaseModel):
    road: str = Field(..., json_schema_extra={"example": "00I24E"}, description="Highway segment corridor code")
    mile: float = Field(..., json_schema_extra={"example": 184.7}, description="Corridor milepost")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of observation (0-23)")
    weather: Optional[str] = Field("Clear", description="Weather condition (Clear, Rain, Fog, etc.)")
    light: Optional[str] = Field("Daylight", description="Lighting condition (Daylight, Dark, Dusk, Dawn)")
    
    # 15-minute aggregate metrics
    speed_mean: float = Field(..., ge=0.0, le=120.0, json_schema_extra={"example": 58.5})
    speed_std: Optional[float] = Field(2.5, ge=0.0)
    speed_min: Optional[float] = Field(None, ge=0.0)
    speed_max: Optional[float] = Field(None, ge=0.0)
    speed_drop: Optional[float] = Field(0.0, description="Speed change over 15 mins (last 5 min avg - first 5 min avg)")
    
    volume_mean: float = Field(..., ge=0.0, json_schema_extra={"example": 12.0})
    volume_std: Optional[float] = Field(3.0, ge=0.0)
    
    occupancy_mean: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 4.5})
    occupancy_max: Optional[float] = Field(None, ge=0.0, le=100.0)
    occupancy_std: Optional[float] = Field(1.5, ge=0.0)
    
    # Spatial neighborhood
    speed_upstream: Optional[float] = Field(None, ge=0.0)
    volume_upstream: Optional[float] = Field(None, ge=0.0)
    occupancy_upstream: Optional[float] = Field(None, ge=0.0)
    
    speed_downstream: Optional[float] = Field(None, ge=0.0)
    volume_downstream: Optional[float] = Field(None, ge=0.0)
    occupancy_downstream: Optional[float] = Field(None, ge=0.0)

    # Optional raw 30-sec observation sequence
    raw_observations: Optional[List[ObservationItem]] = Field(
        None, description="Optional raw sequence of 30-sec observations over 15 mins"
    )

class RiskPredictionResponse(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    road: str
    mile: float
    risk_score: float = Field(..., description="Estimated probability of accident risk in [0, 1]")
    risk_level: str = Field(..., description="Risk category: LOW, MODERATE, HIGH, CRITICAL")
    accident_alert: bool = Field(..., description="True if risk exceeds optimal operational threshold")
    threshold_used: float
    model_version: str
    latency_ms: float
    timestamp: str
