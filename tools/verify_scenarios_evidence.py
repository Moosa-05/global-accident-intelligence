from starlette.testclient import TestClient
from backend.app.main import app
import json

client = TestClient(app)

print("=== 1. HUMAN INJURY SEVERITY PIPELINE ===")
scenarios_injury = [
    ("Scenario 1A: Daytime Motorway Divided 110km/h Dry", {
        "latitude": 51.5074, "longitude": -0.1278, "road_class": "MOTORWAY",
        "speed_limit_kmh": 110, "divided_roadway": True, "light_condition": "DAYLIGHT",
        "weather_condition": "CLEAR", "road_surface_condition": "DRY", "is_junction": False
    }),
    ("Scenario 1B: High-Speed Night Dark Unlit Arterial in Heavy Rain with Junction 120km/h", {
        "latitude": 51.5074, "longitude": -0.1278, "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 120, "divided_roadway": False, "light_condition": "DARKNESS_NO_STREETLIGHTS",
        "weather_condition": "HEAVY_RAIN", "road_surface_condition": "WET_DAMP", "is_junction": True
    }),
    ("Scenario 1C: Low-Speed Urban Residential Non-Junction 30km/h Clear", {
        "latitude": 51.5074, "longitude": -0.1278, "road_class": "LOCAL_RESIDENTIAL",
        "speed_limit_kmh": 30, "divided_roadway": False, "light_condition": "DAYLIGHT",
        "weather_condition": "CLEAR", "road_surface_condition": "DRY", "is_junction": False
    })
]
for title, payload in scenarios_injury:
    res = client.post("/api/v1/predictions/injury-severity", json=payload).json()
    print(f"{title}:")
    print(f"   Payload: road={payload['road_class']}, speed={payload['speed_limit_kmh']}, light={payload['light_condition']}, weather={payload['weather_condition']}")
    print(f"   Response: Severity={res['predicted_severity']}, Probs={res['probabilities']}, FatalAlert={res['fatality_risk_alert']}\n")

print("=== 2. ANNUAL ROAD EXPOSURE RISK PIPELINE ===")
scenarios_exposure = [
    ("Scenario 2A: Low Flow Rural Collector (1.0km, 4,000 AADF, 60km/h)", {
        "road_name": "B100", "road_class": "SECONDARY_COLLECTOR", "link_length_km": 1.0,
        "speed_limit_kmh": 60, "latitude": 53.48, "longitude": -2.24, "annual_average_daily_flow": 4000
    }),
    ("Scenario 2B: Medium Motorway Link (2.5km, 48,500 AADF, 110km/h)", {
        "road_name": "M62", "road_class": "MOTORWAY", "link_length_km": 2.5,
        "speed_limit_kmh": 110, "latitude": 53.48, "longitude": -2.24, "annual_average_daily_flow": 48500
    }),
    ("Scenario 2C: High Volume Major Motorway Corridor (15.0km, 90,000 AADF, 110km/h)", {
        "road_name": "M25", "road_class": "MOTORWAY", "link_length_km": 15.0,
        "speed_limit_kmh": 110, "latitude": 51.50, "longitude": -0.10, "annual_average_daily_flow": 90000
    })
]
for title, payload in scenarios_exposure:
    res = client.post("/api/v1/predictions/exposure-risk", json=payload).json()
    print(f"{title}:")
    print(f"   Payload: AADF={payload['annual_average_daily_flow']}, length={payload['link_length_km']}km, road={payload['road_class']}")
    print(f"   Response: MVKT={res['annual_exposure_million_vkt']}, Crashes/Yr={res['predicted_annual_crashes']}, Rate={res['predicted_crash_rate_per_million_vkt']}/MVKT, Tier={res['risk_tier']}\n")

print("=== 3. CORRIDOR SENSOR RISK PIPELINE ===")
scenarios_corridor = [
    ("Scenario 3A: Free-Flow Off-Peak (65mph, 6 veh/min, 3% occ, Clear)", {
        "road": "00I24E", "mile": 181.5, "speed_mean": 65.0, "speed_drop": 0.0,
        "volume_mean": 6.0, "occupancy_mean": 3.0, "weather": "Clear", "light": "Daylight", "hour": 11
    }),
    ("Scenario 3B: Evening Rush Convective Rain (40mph, 8mph drop, 22 veh/min, 16% occ, Rain)", {
        "road": "00I24E", "mile": 181.5, "speed_mean": 40.0, "speed_drop": 8.0,
        "volume_mean": 22.0, "occupancy_mean": 16.0, "weather": "Rain", "light": "Daylight", "hour": 17
    }),
    ("Scenario 3C: Severe Shockwave Bottleneck (14mph, 28mph drop, 38 veh/min, 58% occ, Snow, Night)", {
        "road": "00I24E", "mile": 181.5, "speed_mean": 14.0, "speed_drop": 28.0,
        "volume_mean": 38.0, "occupancy_mean": 58.0, "weather": "Snow", "light": "Dark", "hour": 22
    })
]
for title, payload in scenarios_corridor:
    res = client.post("/api/v1/predictions/risk", json=payload).json()
    print(f"{title}:")
    print(f"   Payload: speed={payload['speed_mean']}, drop={payload['speed_drop']}, occ={payload['occupancy_mean']}%, weather={payload['weather']}")
    print(f"   Response: RiskScore={res['risk_score']} ({(res['risk_score']*100):.2f}%), Level={res['risk_level']}, Alert={res['accident_alert']}\n")

print("=== 4. TRAFFIC IMPACT SEVERITY PIPELINE ===")
scenarios_severity = [
    ("Scenario 4A: Mild Weather Highway Junction (LA, 75F, Clear, Junction)", {
        "latitude": 34.05, "longitude": -118.24, "units": "imperial", "temperature": 75.0,
        "humidity": 40.0, "pressure": 29.92, "visibility": 10.0, "wind_speed": 6.0,
        "weather_condition": "Clear", "is_junction": True, "is_traffic_signal": True
    }),
    ("Scenario 4B: Severe Rain & Low Pressure Corridor (Chicago, 45F, Heavy Rain, Signal)", {
        "latitude": 41.87, "longitude": -87.62, "units": "imperial", "temperature": 45.0,
        "humidity": 92.0, "pressure": 29.30, "visibility": 3.0, "wind_speed": 25.0,
        "precipitation": 0.8, "weather_condition": "Rain", "is_junction": True, "is_traffic_signal": True
    }),
    ("Scenario 4C: Freezing Snow Non-Junction (Seattle, 26F, Snow, Non-Junction)", {
        "latitude": 47.60, "longitude": -122.33, "units": "imperial", "temperature": 26.0,
        "humidity": 88.0, "pressure": 29.80, "visibility": 1.5, "wind_speed": 18.0,
        "precipitation": 0.4, "weather_condition": "Snow", "is_junction": False, "is_traffic_signal": False
    })
]
for title, payload in scenarios_severity:
    res = client.post("/api/v1/predictions/severity", json=payload).json()
    print(f"{title}:")
    print(f"   Payload: temp={payload['temperature']}F, precip={payload.get('precipitation', 0)}, weather={payload['weather_condition']}")
    print(f"   Response: DelayClass={res['predicted_severity']}, Level={res['severity_level']}, Probs={res['severity_probabilities']}\n")
