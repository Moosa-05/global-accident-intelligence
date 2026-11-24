from enum import Enum
from typing import Optional

class UnitSystem(str, Enum):
    METRIC = "metric"
    IMPERIAL = "imperial"

class UnitConverter:
    """
    Standardized, bidirectional unit converter ensuring the global SaaS platform
    can accept inputs from any country or measurement system and normalize them
    to canonical metric units for machine learning inference.
    """

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32.0) * (5.0 / 9.0)

    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return (c * 9.0 / 5.0) + 32.0

    @staticmethod
    def mph_to_kmh(mph: float) -> float:
        return mph * 1.609344

    @staticmethod
    def kmh_to_mph(kmh: float) -> float:
        return kmh / 1.609344

    @staticmethod
    def inhg_to_hpa(inhg: float) -> float:
        return inhg * 33.863886666667

    @staticmethod
    def hpa_to_inhg(hpa: float) -> float:
        return hpa / 33.863886666667

    @staticmethod
    def miles_to_km(miles: float) -> float:
        return miles * 1.609344

    @staticmethod
    def km_to_miles(km: float) -> float:
        return km / 1.609344

    @staticmethod
    def inches_to_mm(inches: float) -> float:
        return inches * 25.4

    @staticmethod
    def mm_to_inches(mm: float) -> float:
        return mm / 25.4

    @classmethod
    def normalize_weather_inputs(
        cls,
        temperature: float,
        wind_speed: float,
        visibility: float,
        pressure: float,
        precipitation: float,
        input_units: UnitSystem = UnitSystem.METRIC
    ) -> dict:
        """
        Converts incoming weather telemetry to canonical metric values:
          - temperature_c: Celsius
          - wind_speed_kmh: km/h
          - visibility_km: km
          - pressure_hpa: hPa / mbar
          - precipitation_mm: mm
        """
        if input_units == UnitSystem.IMPERIAL:
            return {
                "temperature_c": cls.fahrenheit_to_celsius(temperature),
                "wind_speed_kmh": cls.mph_to_kmh(wind_speed),
                "visibility_km": cls.miles_to_km(visibility),
                "pressure_hpa": cls.inhg_to_hpa(pressure),
                "precipitation_mm": cls.inches_to_mm(precipitation),
                "source_units": "imperial"
            }
        return {
            "temperature_c": float(temperature),
            "wind_speed_kmh": float(wind_speed),
            "visibility_km": float(visibility),
            "pressure_hpa": float(pressure),
            "precipitation_mm": float(precipitation),
            "source_units": "metric"
        }
