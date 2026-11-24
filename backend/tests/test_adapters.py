from backend.app.adapters.weather_adapter import GenericWeatherAdapter
from backend.app.adapters.telematics_adapter import TelematicsAdapter
from backend.app.adapters.road_geometry_adapter import RoadGeometryAdapter

def test_weather_adapter():
    adapter = GenericWeatherAdapter()
    res = adapter.adapt({
        "temperature": 18.0,
        "wind_speed": 12.0,
        "visibility": 8.0,
        "pressure": 1015.0,
        "precipitation": 1.5,
        "condition": "Scattered Thunderstorms",
        "units": "metric"
    })
    assert res["temperature_c"] == 18.0
    assert res["weather_condition"] == "Thunderstorm"
    assert res["weather_code"] == 5

def test_telematics_adapter():
    adapter = TelematicsAdapter()
    res = adapter.adapt({
        "latitude": 51.5074,
        "longitude": -0.1278,
        "speed": 60.0,
        "heading": 180.0,
        "units": "imperial",
        "timestamp": "2026-11-19T14:00:00Z"
    })
    assert res["latitude"] == 51.5074
    assert res["longitude"] == -0.1278
    assert round(res["speed_kmh"], 1) == 96.6
    assert res["hour"] == 14

def test_road_geometry_adapter():
    adapter = RoadGeometryAdapter()
    res = adapter.adapt({
        "tags": {
            "highway": "motorway_junction",
            "traffic_signals": "yes",
            "junction": "roundabout"
        }
    })
    assert res["is_junction"] is True
    assert res["is_traffic_signal"] is True
    assert res["is_roundabout"] is True
    assert res["is_railway"] is False
