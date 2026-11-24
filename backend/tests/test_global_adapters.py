"""Unit tests for Phase 4 Global Data Adapters and Canonical Schema"""
import pytest
from pathlib import Path
import polars as pl

from backend.app.adapters.uk_stats19_adapter import UKStats19Adapter
from backend.app.adapters.france_baac_adapter import FranceBAACAdapter
from backend.app.adapters.era5_weather_adapter import ERA5WeatherAdapter
from backend.app.schemas.canonical_accident import (
    CountryCode,
    LicenseType,
    HumanInjurySeverity,
    CanonicalRoadClass
)

def test_uk_stats19_adapter_sample():
    adapter = UKStats19Adapter()
    df = adapter.ingest(limit=100)
    assert len(df) == 100
    assert "global_accident_id" in df.columns
    assert "human_injury_severity" in df.columns
    assert "traffic_impact_severity" in df.columns
    assert df["country_code"][0] == CountryCode.GB.value
    assert df["licence_type"][0] == LicenseType.OGL_V3.value
    
    # Severity must be exclusively HumanInjurySeverity (not numbers 1-4)
    severities = set(df["human_injury_severity"].to_list())
    for s in severities:
        assert s in [HumanInjurySeverity.SLIGHT.value, HumanInjurySeverity.SERIOUS.value, HumanInjurySeverity.FATAL.value]
    
    # Traffic impact severity must be null (not monitored in STATS19)
    assert df["traffic_impact_severity"].null_count() == 100
    
    # Coordinates must be valid WGS84
    for lat, lon in zip(df["latitude"], df["longitude"]):
        assert 49.0 <= lat <= 61.0
        assert -8.5 <= lon <= 2.5

def test_france_baac_adapter_sample():
    adapter = FranceBAACAdapter()
    df = adapter.ingest(limit=100)
    assert len(df) == 100
    assert "global_accident_id" in df.columns
    assert df["country_code"][0] == CountryCode.FR.value
    assert df["licence_type"][0] == LicenseType.LICENCE_OUVERTE_V2.value
    
    # Verify decimal comma parsing to float
    for lat, lon in zip(df["latitude"], df["longitude"]):
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0

    # Severity values
    severities = set(df["human_injury_severity"].to_list())
    for s in severities:
        assert s in [HumanInjurySeverity.SLIGHT.value, HumanInjurySeverity.SERIOUS.value, HumanInjurySeverity.FATAL.value, HumanInjurySeverity.UNINJURED.value]

def test_era5_weather_adapter_fallback():
    adapter = ERA5WeatherAdapter()
    # Test fallback / fetch
    obs = adapter.fetch_observations(lat=51.5, lon=-0.1, date_str="2022-09-23", hour=14)
    assert "temperature_c" in obs
    assert "humidity_pct" in obs
    assert "pressure_hpa" in obs
    assert "wind_speed_kmh" in obs
    assert "precipitation_mm" in obs
    assert isinstance(obs["temperature_c"], float)
