import pytest
from backend.app.core.units import UnitConverter, UnitSystem

def test_temperature_conversions():
    assert UnitConverter.fahrenheit_to_celsius(32.0) == 0.0
    assert round(UnitConverter.fahrenheit_to_celsius(212.0), 1) == 100.0
    assert UnitConverter.celsius_to_fahrenheit(0.0) == 32.0
    assert UnitConverter.celsius_to_fahrenheit(100.0) == 212.0

def test_speed_conversions():
    assert round(UnitConverter.mph_to_kmh(60.0), 2) == 96.56
    assert round(UnitConverter.kmh_to_mph(100.0), 2) == 62.14

def test_pressure_conversions():
    # 29.92 inHg ~ 1013.2 hPa
    assert round(UnitConverter.inhg_to_hpa(29.92), 1) == 1013.2

def test_distance_and_precipitation():
    assert round(UnitConverter.miles_to_km(10.0), 2) == 16.09
    assert round(UnitConverter.inches_to_mm(1.0), 1) == 25.4

def test_normalize_weather_inputs_imperial():
    norm = UnitConverter.normalize_weather_inputs(
        temperature=77.0,     # 25 C
        wind_speed=15.0,     # ~24.1 km/h
        visibility=10.0,     # ~16.1 km
        pressure=29.92,      # ~1013.2 hPa
        precipitation=0.5,   # 12.7 mm
        input_units=UnitSystem.IMPERIAL
    )
    assert round(norm["temperature_c"], 1) == 25.0
    assert round(norm["precipitation_mm"], 1) == 12.7
    assert norm["source_units"] == "imperial"
