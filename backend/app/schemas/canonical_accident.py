"""Canonical Country-Independent Accident Data Schema.
Defines the standard cross-country representation for road safety events globally.
Complies with WGS84 coordinates, ISO-8601 UTC timestamps, universal physical units,
and strictly separates Traffic Impact Severity from Human Injury Severity.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class CountryCode(str, Enum):
    GB = "GB"
    FR = "FR"
    US = "US"
    DE = "DE"
    AU = "AU"
    CA = "CA"
    OTHER = "OTHER"

class LicenseType(str, Enum):
    OGL_V3 = "Open Government Licence v3.0 (UK)"
    LICENCE_OUVERTE_V2 = "Licence Ouverte / Open Licence v2.0 (France)"
    DL_DE_BY_2_0 = "Datenlizenz Deutschland – Namensnennung – Version 2.0 (Germany)"
    CC_BY_4_0 = "Creative Commons Attribution 4.0 (Australia / ERA5)"
    OGL_CANADA = "Open Government Licence – Canada"
    ODBL_V1 = "Open Data Commons Open Database License (OSM)"
    CC_BY_NC_SA_4_0 = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (US Accidents)"
    PUBLIC_DOMAIN = "Public Domain"

class CanonicalRoadClass(str, Enum):
    MOTORWAY = "MOTORWAY"               # UK M-roads, FR Autoroutes, US Interstates
    TRUNK_HIGHWAY = "TRUNK_HIGHWAY"     # UK A(M)/Major Trunk, FR Nationales, US US-Routes
    PRIMARY_ARTERIAL = "PRIMARY_ARTERIAL" # UK A-roads, FR Départementales principales, US State Routes
    SECONDARY_COLLECTOR = "SECONDARY_COLLECTOR" # UK B-roads, FR Départementales, US County Highways
    TERTIARY_ROAD = "TERTIARY_ROAD"     # UK C-roads, FR Voies intercommunales
    LOCAL_RESIDENTIAL = "LOCAL_RESIDENTIAL" # Residential streets, urban distribution
    LOCAL_RURAL = "LOCAL_RURAL"         # Unclassified rural roads
    SERVICE_ACCESS = "SERVICE_ACCESS"   # Service alleys, parking access

class RoadSurfaceCondition(str, Enum):
    DRY = "DRY"
    WET_DAMP = "WET_DAMP"
    SNOW_SLUSH = "SNOW_SLUSH"
    FROST_ICE = "FROST_ICE"
    FLOOD = "FLOOD"
    MUD_OIL = "MUD_OIL"
    UNKNOWN = "UNKNOWN"

class LightCondition(str, Enum):
    DAYLIGHT = "DAYLIGHT"
    TWILIGHT_DAWN_DUSK = "TWILIGHT_DAWN_DUSK"
    DARKNESS_STREETLIGHTS_ON = "DARKNESS_STREETLIGHTS_ON"
    DARKNESS_STREETLIGHTS_OFF = "DARKNESS_STREETLIGHTS_OFF"
    DARKNESS_NO_STREETLIGHTS = "DARKNESS_NO_STREETLIGHTS"
    UNKNOWN = "UNKNOWN"

class WeatherConditionCategory(str, Enum):
    CLEAR = "CLEAR"
    CLOUDY_OVERCAST = "CLOUDY_OVERCAST"
    LIGHT_RAIN = "LIGHT_RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"
    SNOW_HAIL = "SNOW_HAIL"
    FOG_SMOKE = "FOG_SMOKE"
    HIGH_WINDS_STORM = "HIGH_WINDS_STORM"
    OTHER = "OTHER"

class HumanInjurySeverity(str, Enum):
    """Human physiological injury outcome (UNECE / WHO standard 30-day definition).
    NEVER merge with traffic impact delay severity.
    """
    UNINJURED = "UNINJURED"     # No injury recorded (damage only / indemnity)
    SLIGHT = "SLIGHT"           # Minor injury, no inpatient hospital stay > 24h
    SERIOUS = "SERIOUS"         # Severe trauma, hospitalization > 24h
    FATAL = "FATAL"             # Death within 30 days of the collision
    UNKNOWN = "UNKNOWN"

class TrafficImpactSeverity(int, Enum):
    """Traffic congestion & network disruption outcome (US DoT benchmark delay scale).
    NEVER merge with human physiological injury severity.
    """
    MINIMAL_IMPACT = 1          # Short delay < 15 min, lane open
    MODERATE_IMPACT = 2         # Delay 15-45 min, standard blockage
    SIGNIFICANT_IMPACT = 3      # Delay 45-90 min, multi-lane obstruction
    SEVERE_DISRUPTION = 4       # Delay > 90 min, corridor closure

class CanonicalAccidentRecord(BaseModel):
    # Global Identifiers & Provenance
    global_accident_id: str = Field(..., description="Globally unique canonical accident ID (e.g. GB-STATS19-202201...)")
    source_record_id: str = Field(..., description="Original record ID from the national data source")
    source_dataset: str = Field(..., description="Name of national reporting dataset")
    country_code: CountryCode = Field(..., description="ISO 3166-1 alpha-2 country code")
    licence_type: LicenseType = Field(..., description="Verified open data licence type")
    attribution_text: str = Field(..., description="Required legal attribution string")

    # Spatial Location (WGS84 EPSG:4326)
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84 Latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84 Longitude")
    coordinate_precision: str = Field("HIGH_GPS", description="Precision of coordinate resolution")
    administrative_region: Optional[str] = None
    locality_or_city: Optional[str] = None

    # Temporal & Astronomical (ISO-8601 UTC)
    timestamp_utc: str = Field(..., description="UTC timestamp in ISO-8601 format (YYYY-MM-DDTHH:MM:SSZ)")
    local_timestamp: str = Field(..., description="Local solar/clock timestamp (YYYY-MM-DDTHH:MM:SS)")
    timezone_iana: str = Field(..., description="IANA timezone identifier (e.g. 'Europe/London', 'Europe/Paris')")
    year: int
    month: int
    day: int
    hour: int
    minute: int
    day_of_week: int = Field(..., ge=1, le=7, description="1=Monday ... 7=Sunday")
    is_weekend: bool
    is_night: bool

    # Roadway & Geometric Characteristics
    road_class: CanonicalRoadClass
    speed_limit_kmh: Optional[float] = None
    is_junction: bool = False
    is_traffic_signal: bool = False
    is_crossing: bool = False
    is_roundabout: bool = False
    is_stop_sign: bool = False
    is_give_way: bool = False
    is_railway: bool = False
    is_traffic_calming: bool = False
    number_of_lanes: Optional[int] = None
    divided_roadway: Optional[bool] = None

    # Environmental & Atmospheric (Canonical Metric)
    weather_condition: WeatherConditionCategory
    light_condition: LightCondition
    road_surface_condition: RoadSurfaceCondition
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    visibility_km: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    precipitation_mm: Optional[float] = None

    # Vehicles & Casualties
    num_vehicles_involved: int = Field(1, ge=1)
    num_casualties_total: int = Field(0, ge=0)
    casualty_fatal_count: int = Field(0, ge=0)
    casualty_serious_count: int = Field(0, ge=0)
    casualty_slight_count: int = Field(0, ge=0)

    # STRICT SEPARATION OF TARGETS
    human_injury_severity: HumanInjurySeverity = Field(
        ...,
        description="Physical human trauma severity: UNINJURED, SLIGHT, SERIOUS, FATAL"
    )
    traffic_impact_severity: Optional[TrafficImpactSeverity] = Field(
        None,
        description="Traffic network congestion delay severity: 1, 2, 3, 4 (null if not monitored by telematics)"
    )

    # Traffic Exposure Join Fields
    annual_average_daily_flow: Optional[int] = Field(None, description="Nearby traffic volume count (AADF) if joined")
    distance_to_traffic_counter_km: Optional[float] = None
