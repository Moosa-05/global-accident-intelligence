"""Pydantic schemas for Exposure-Adjusted Road Segment Crash Frequency & Rate Prediction.
Models annual crash frequency and crash rates per Million Vehicle-Kilometres Travelled (MVKT)
using Poisson / Negative Binomial GLMs with log-exposure offsets.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid

class ExposureRiskPredictionRequest(BaseModel):
    # Segment Identification & Geometry
    road_name: Optional[str] = Field("A1", description="Road name or route identifier")
    road_class: str = Field("MOTORWAY", description="Road hierarchy: 'MOTORWAY', 'TRUNK_HIGHWAY', 'PRIMARY_ARTERIAL', 'SECONDARY_COLLECTOR', 'LOCAL_OTHER'")
    link_length_km: float = Field(..., gt=0.0, le=100.0, description="Length of the monitored road link in kilometres")
    speed_limit_kmh: float = Field(100.0, ge=20.0, le=150.0, description="Regulatory speed limit in km/h")
    
    # Location
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84 Latitude of segment midpoint")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84 Longitude of segment midpoint")
    
    # Traffic Exposure
    annual_average_daily_flow: int = Field(..., gt=0, description="Annual Average Daily Flow (AADF / TMJA) in vehicles per day")

class ExposureRiskPredictionResponse(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    annual_exposure_million_vkt: float = Field(..., description="Annual Vehicle-Kilometres Travelled (AADF * 365 * length / 1e6)")
    predicted_annual_crashes: float = Field(..., description="Expected annual injury collision count on this link (E[Y])")
    predicted_crash_rate_per_million_vkt: float = Field(..., description="Normalized crash rate: predicted crashes per Million Vehicle-Kilometres (MVKT)")
    risk_tier: str = Field(..., description="'LOW_RISK', 'TYPICAL_RISK', 'ELEVATED_RISK', or 'HIGH_RISK'")
    relative_risk_to_national_baseline: float = Field(..., description="Ratio of segment predicted crash rate to national baseline motorway/arterial rate")
    statistical_model: str = "Negative Binomial GLM with Log-Exposure Offset"
    model_version: str = "1.0.0"
    exposure_interpretation_notice: str = (
        "Macro-scale annual road segment model. Represents annual expected crash frequency based on cumulative "
        "traffic flow exposure. Strictly separate from instantaneous micro-scale hourly congestion risk."
    )
    latency_ms: float
    timestamp: str
