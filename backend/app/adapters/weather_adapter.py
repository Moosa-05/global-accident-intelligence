from typing import Dict, Any
from backend.app.adapters.base import BaseDataProviderAdapter
from backend.app.core.units import UnitConverter, UnitSystem

class GenericWeatherAdapter(BaseDataProviderAdapter):
    """
    Normalizes third-party weather data (e.g., OpenWeatherMap, MeteoBlue, NOAA)
    into the canonical global weather schema.
    """

    WEATHER_KEYWORD_MAP = {
        "clear": "Clear",
        "sun": "Clear",
        "fair": "Clear",
        "cloud": "Cloudy",
        "overcast": "Cloudy",
        "rain": "Rain",
        "drizzle": "Rain",
        "shower": "Rain",
        "snow": "Snow",
        "ice": "Snow",
        "sleet": "Snow",
        "blizzard": "Snow",
        "fog": "Fog",
        "mist": "Fog",
        "haze": "Fog",
        "smoke": "Fog",
        "thunder": "Thunderstorm",
        "storm": "Thunderstorm",
    }

    def map_condition(self, raw_condition: str) -> str:
        c_lower = str(raw_condition).lower()
        for kw, cat in self.WEATHER_KEYWORD_MAP.items():
            if kw in c_lower:
                return cat
        return "Clear"

    def map_category_to_code(self, category: str) -> int:
        codes = {
            "Clear": 0,
            "Cloudy": 1,
            "Rain": 2,
            "Snow": 3,
            "Fog": 4,
            "Thunderstorm": 5,
            "Other": 6
        }
        return codes.get(category, 0)

    def adapt(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        unit_system = UnitSystem(raw_payload.get("units", "metric").lower())
        temp = float(raw_payload.get("temperature", 20.0))
        wind = float(raw_payload.get("wind_speed", 10.0))
        vis = float(raw_payload.get("visibility", 10.0))
        press = float(raw_payload.get("pressure", 1013.25))
        precip = float(raw_payload.get("precipitation", 0.0))
        cond = self.map_condition(raw_payload.get("condition", "Clear"))

        normalized = UnitConverter.normalize_weather_inputs(
            temperature=temp,
            wind_speed=wind,
            visibility=vis,
            pressure=press,
            precipitation=precip,
            input_units=unit_system
        )
        normalized["weather_condition"] = cond
        normalized["weather_code"] = self.map_category_to_code(cond)
        return normalized
