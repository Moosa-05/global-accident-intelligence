import urllib.request
import json

LOCATIONS = [
    ('Lahore, Pakistan', 31.5204, 74.3587),
    ('London, United Kingdom', 51.5074, -0.1278),
    ('Paris, France', 48.8566, 2.3522),
    ('Berlin, Germany', 52.5200, 13.4050),
    ('Tokyo, Japan', 35.6762, 139.6503),
    ('New York, USA', 40.7128, -74.0060),
    ('Sydney, Australia', -33.8688, 151.2093),
    ('Scottish Highlands (Rural UK)', 57.3229, -4.4244),
    ('Outback (Remote Australia)', -25.2744, 133.7751)
]

print("=" * 90)
print("1. HUMAN INJURY SEVERITY PIPELINE: GEOGRAPHICAL EVALUATION")
print("=" * 90)
url_inj = "http://127.0.0.1:8000/api/v1/predictions/injury-severity"
for name, lat, lng in LOCATIONS:
    payload = {
        "latitude": lat,
        "longitude": lng,
        "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 80,
        "weather_condition": "CLEAR",
        "light_condition": "DAYLIGHT",
        "road_surface_condition": "DRY",
        "is_junction": False,
        "divided_roadway": True
    }
    req = urllib.request.Request(url_inj, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        geo = res.get("geographic_metadata", {})
        p = res["probabilities"]
        print(f"{name:30s} | Pred: {res['predicted_severity']:7s} | Slight: {p['SLIGHT']:.3f} | Serious: {p['SERIOUS']:.3f} | Fatal: {p['FATAL']:.3f} | OOD: {str(geo.get('is_out_of_distribution')):5s} | Status: {geo.get('validation_status')}")

print("\n" + "=" * 90)
print("2. ANNUAL ROAD EXPOSURE RISK PIPELINE: GEOGRAPHICAL EVALUATION")
print("=" * 90)
url_exp = "http://127.0.0.1:8000/api/v1/predictions/exposure-risk"
for name, lat, lng in LOCATIONS:
    payload = {
        "latitude": lat,
        "longitude": lng,
        "road_class": "MOTORWAY",
        "road_name": f"{name} Link",
        "speed_limit_kmh": 110,
        "link_length_km": 2.5,
        "annual_average_daily_flow": 45000
    }
    req = urllib.request.Request(url_exp, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"{name:30s} | MVKT: {res['annual_exposure_million_vkt']:6.2f} | Crashes/Yr: {res['predicted_annual_crashes']:5.2f} | Rate: {res['predicted_crash_rate_per_million_vkt']:.4f}/MVKT | Tier: {res['risk_tier']}")

print("\n" + "=" * 90)
print("3. TRAFFIC IMPACT SEVERITY PIPELINE: GEOGRAPHICAL EVALUATION")
print("=" * 90)
url_sev = "http://127.0.0.1:8000/api/v1/predictions/severity"
for name, lat, lng in LOCATIONS:
    payload = {
        "latitude": lat,
        "longitude": lng,
        "units": "metric",
        "temperature": 20.0,
        "humidity": 60.0,
        "pressure": 1013.25,
        "visibility": 10.0,
        "wind_speed": 10.0,
        "precipitation": 0.0,
        "weather_condition": "Clear",
        "is_junction": True,
        "is_traffic_signal": True
    }
    req = urllib.request.Request(url_sev, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        geo = res.get("geographic_metadata", {})
        probs = res["severity_probabilities"]
        print(f"{name:30s} | Class: {res['predicted_severity']} ({res['severity_level']:18s}) | C1: {probs['class_1']:.3f} C2: {probs['class_2']:.3f} C3: {probs['class_3']:.3f} C4: {probs['class_4']:.3f} | OOD: {str(geo.get('is_out_of_distribution')):5s} | Status: {geo.get('validation_status')}")

print("\n" + "=" * 90)
print("4. CORRIDOR SENSOR RISK PIPELINE: TELEMETRY EVALUATION")
print("=" * 90)
url_risk = "http://127.0.0.1:8000/api/v1/predictions/risk"
SCENARIOS_RISK = [
    ("Free-Flow Off-Peak", "00I24E", 181.5, 65.0, 0.0, 8.0, 3.0, "Clear", "Daylight", 14),
    ("Moderate Congestion", "00I24E", 181.5, 42.0, 0.0, 18.5, 12.0, "Rain", "Daylight", 17),
    ("Approaching Shockwave", "00I24E", 181.5, 30.0, 20.0, 28.0, 35.0, "Rain", "Dark", 19),
    ("Severe Deceleration Jam", "00I24E", 181.5, 18.0, 35.0, 38.0, 60.0, "Snow", "Dark", 21),
    ("I-75 Northbound Split", "00I75N", 2.0, 55.0, 5.0, 22.0, 14.0, "Clear", "Daylight", 16),
    ("SR-29 Winter Corridor", "0SR29N", 5.0, 22.0, 25.0, 32.0, 50.0, "Snow", "Dark", 22)
]
for title, rd, mi, spd, drp, vol, occ, wtr, lgt, hr in SCENARIOS_RISK:
    payload = {
        "road": rd,
        "mile": mi,
        "speed_mean": spd,
        "speed_drop": drp,
        "volume_mean": vol,
        "occupancy_mean": occ,
        "weather": wtr,
        "light": lgt,
        "hour": hr
    }
    req = urllib.request.Request(url_risk, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"{title:25s} | Road: {res['road']:6s} Mile: {res['mile']:5.1f} | Risk: {res['risk_score']*100:5.2f}% | Level: {res['risk_level']:8s} | Alert: {str(res['accident_alert']):5s}")
