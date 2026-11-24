"""Human Injury Severity Prediction Service (3-Class).
Inference service for collision-level physiological trauma severity:
- SLIGHT
- SERIOUS
- FATAL
"""
import sys
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.units import UnitConverter, UnitSystem
from backend.app.schemas.severity import GeographicMetadata
from backend.app.schemas.injury_severity import (
    InjurySeverityPredictionRequest,
    InjurySeverityPredictionResponse
)
from backend.app.db.models.injury_prediction_log import InjuryPredictionLog

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SEVERITY_NAMES = {0: "SLIGHT", 1: "SERIOUS", 2: "FATAL"}

class InjurySeverityService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.schema = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        models_dir = PROJECT_ROOT / "models"
        model_path = models_dir / "human_injury_severity_lgbm.pkl"
        pipe_path = models_dir / "injury_feature_pipeline.pkl"
        meta_path = models_dir / "injury_model_metadata.json"

        try:
            if model_path.exists() and pipe_path.exists():
                self.model = joblib.load(model_path)
                self.preprocessor = joblib.load(pipe_path)
                if meta_path.exists():
                    import json
                    with open(meta_path, "r") as f:
                        self.metadata = json.load(f)
                logger.info("Human Injury Severity model and preprocessor loaded successfully.")
            else:
                logger.warning(f"Injury severity model artifacts not found at {model_path}.")
        except Exception as e:
            logger.error(f"Failed to load injury severity model artifacts: {e}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    @property
    def model_version(self) -> str:
        if self.metadata and "version" in self.metadata:
            return self.metadata["version"]
        return "1.0.0"

    def _check_geographic_coverage(self, lat: float, lng: float, timezone_str: str) -> GeographicMetadata:
        # Validated training footprint: UK & France (Western Europe)
        # 41N - 60N, -8W - 10E
        in_eur = (41.0 <= lat <= 61.0) and (-9.0 <= lng <= 10.5)
        
        if in_eur:
            return GeographicMetadata(
                coordinates={"latitude": lat, "longitude": lng},
                timezone=timezone_str,
                is_out_of_distribution=False,
                validation_status="EMPIRICALLY_VALIDATED",
                validation_warning=None,
                training_geography="United Kingdom & France (STATS19 & BAAC Benchmark)",
                licensing_terms="Open Government Licence v3.0 & Licence Ouverte v2.0 (Commercial Cleared)"
            )
        else:
            return GeographicMetadata(
                coordinates={"latitude": lat, "longitude": lng},
                timezone=timezone_str,
                is_out_of_distribution=True,
                validation_status="UNVALIDATED_TERRITORY",
                validation_warning=(
                    f"Location ({lat}, {lng}) is outside the empirical training coverage (UK & France road safety corpora). "
                    f"Model predictions reflect atmospheric and geometric heuristics without localized road baseline calibration."
                ),
                training_geography="United Kingdom & France (STATS19 & BAAC Benchmark)",
                licensing_terms="Open Government Licence v3.0 & Licence Ouverte v2.0 (Commercial Cleared)"
            )

    def predict(self, req: InjurySeverityPredictionRequest, db_session = None, organization_id: Optional[str] = None) -> InjurySeverityPredictionResponse:
        t0 = time.time()
        
        # 1. Temporal handling
        dt = req.timestamp or datetime.utcnow()
        hour = dt.hour
        month = dt.month
        day_of_week = dt.isoweekday()
        is_weekend = day_of_week in (6, 7)
        
        hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
        month_sin = math.sin(2.0 * math.pi * (month - 1) / 12.0)
        month_cos = math.cos(2.0 * math.pi * (month - 1) / 12.0)

        # 2. Unit conversion
        if req.units == UnitSystem.IMPERIAL:
            temp_c = UnitConverter.fahrenheit_to_celsius(req.temperature) if req.temperature is not None else 15.0
            wind_kmh = UnitConverter.mph_to_kmh(req.wind_speed) if req.wind_speed is not None else 15.0
            pres_hpa = UnitConverter.inhg_to_hpa(req.pressure) if req.pressure is not None else 1013.25
            precip_mm = UnitConverter.inches_to_mm(req.precipitation) if req.precipitation is not None else 0.0
        else:
            temp_c = req.temperature if req.temperature is not None else 15.0
            wind_kmh = req.wind_speed if req.wind_speed is not None else 15.0
            pres_hpa = req.pressure if req.pressure is not None else 1013.25
            precip_mm = req.precipitation if req.precipitation is not None else 0.0

        # 3. Canonicalize categories and resolve aliases
        # Road Class
        road_cls = str(req.road_class or "PRIMARY_ARTERIAL").upper()
        if road_cls in ["A_ROAD", "PRIMARY", "TRUNK"]:
            road_cls = "PRIMARY_ARTERIAL"
        elif road_cls in ["B_ROAD", "SECONDARY"]:
            road_cls = "SECONDARY_COLLECTOR"
        elif road_cls in ["LOCAL_ROAD", "LOCAL", "URBAN"]:
            road_cls = "LOCAL_RESIDENTIAL"
        elif road_cls in ["RURAL", "UNCLASSIFIED"]:
            road_cls = "LOCAL_RURAL"
        elif road_cls in ["MOTORWAY", "AUTOROUTE", "HIGHWAY"]:
            road_cls = "MOTORWAY"
        elif road_cls not in ["LOCAL_RESIDENTIAL", "LOCAL_RURAL", "MOTORWAY", "PRIMARY_ARTERIAL", "SECONDARY_COLLECTOR", "SERVICE_ACCESS", "TERTIARY_ROAD", "TRUNK_HIGHWAY"]:
            road_cls = "PRIMARY_ARTERIAL"

        # Light condition & night flag
        raw_light = str(req.light_condition or req.light_conditions or "DAYLIGHT").upper()
        if "DARK_LIGHTS_LIT" in raw_light or "STREETLIGHTS_ON" in raw_light:
            light_cond = "DARKNESS_STREETLIGHTS_ON"
            is_night_flag = True
        elif "DARK_NO_LIGHTS" in raw_light or "NO_STREETLIGHTS" in raw_light:
            light_cond = "DARKNESS_NO_STREETLIGHTS"
            is_night_flag = True
        elif "DARK" in raw_light:
            light_cond = "DARKNESS_NO_STREETLIGHTS"
            is_night_flag = True
        elif "TWILIGHT" in raw_light or "DUSK" in raw_light or "DAWN" in raw_light:
            light_cond = "TWILIGHT_DAWN_DUSK"
            is_night_flag = False
        elif raw_light in ["DARKNESS_NO_STREETLIGHTS", "DARKNESS_STREETLIGHTS_OFF", "DARKNESS_STREETLIGHTS_ON", "DAYLIGHT", "TWILIGHT_DAWN_DUSK", "UNKNOWN"]:
            light_cond = raw_light
            is_night_flag = "DARK" in raw_light
        else:
            light_cond = "DAYLIGHT"
            is_night_flag = False

        if req.is_night is not None:
            is_night_flag = bool(req.is_night)

        # Weather condition
        raw_weather = str(req.weather_condition or req.weather_conditions or "CLEAR").upper()
        if "FINE" in raw_weather or "CLEAR" in raw_weather:
            weather_cond = "CLEAR"
        elif "CLOUDY" in raw_weather or "OVERCAST" in raw_weather:
            weather_cond = "CLOUDY_OVERCAST"
        elif "HEAVY_RAIN" in raw_weather or "STORM" in raw_weather:
            weather_cond = "HEAVY_RAIN"
        elif "RAIN" in raw_weather or "DRIZZLE" in raw_weather or "WET" in raw_weather:
            weather_cond = "LIGHT_RAIN"
        elif "SNOW" in raw_weather or "SLEET" in raw_weather or "HAIL" in raw_weather:
            weather_cond = "SNOW_HAIL"
        elif "FOG" in raw_weather or "MIST" in raw_weather or "SMOKE" in raw_weather:
            weather_cond = "FOG_SMOKE"
        elif "WIND" in raw_weather:
            weather_cond = "HIGH_WINDS_STORM"
        elif raw_weather in ["CLEAR", "CLOUDY_OVERCAST", "FOG_SMOKE", "HEAVY_RAIN", "HIGH_WINDS_STORM", "LIGHT_RAIN", "OTHER", "SNOW_HAIL"]:
            weather_cond = raw_weather
        else:
            weather_cond = "CLEAR"

        # Road surface condition
        raw_surface = str(req.road_surface_condition or req.surface_conditions or "DRY").upper()
        if "WET" in raw_surface or "DAMP" in raw_surface:
            surface_cond = "WET_DAMP"
        elif "SNOW" in raw_surface or "SLUSH" in raw_surface:
            surface_cond = "SNOW_SLUSH"
        elif "ICE" in raw_surface or "FROST" in raw_surface:
            surface_cond = "FROST_ICE"
        elif "FLOOD" in raw_surface:
            surface_cond = "FLOOD"
        elif raw_surface in ["DRY", "FLOOD", "FROST_ICE", "SNOW_SLUSH", "UNKNOWN", "WET_DAMP"]:
            surface_cond = raw_surface
        else:
            surface_cond = "DRY"

        divided = bool(req.divided_roadway)
        if road_cls == "MOTORWAY":
            divided = True

        # 4. Assemble raw features DataFrame
        features_dict = {
            "latitude": [req.latitude],
            "longitude": [req.longitude],
            "speed_limit_kmh": [req.speed_limit_kmh or 50.0],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "month_sin": [month_sin],
            "month_cos": [month_cos],
            "day_of_week": [day_of_week],
            "temperature_c": [temp_c],
            "humidity_pct": [req.humidity or 70.0],
            "pressure_hpa": [pres_hpa],
            "wind_speed_kmh": [wind_kmh],
            "precipitation_mm": [precip_mm],
            "is_junction": [bool(req.is_junction)],
            "is_traffic_signal": [bool(req.is_traffic_signal)],
            "is_crossing": [bool(req.is_crossing)],
            "is_roundabout": [bool(req.is_roundabout)],
            "is_stop_sign": [bool(req.is_stop_sign)],
            "is_give_way": [bool(req.is_give_way)],
            "is_railway": [bool(req.is_railway)],
            "is_weekend": [is_weekend],
            "is_night": [is_night_flag],
            "divided_roadway": [divided],
            "road_class": [road_cls],
            "weather_condition": [weather_cond],
            "light_condition": [light_cond],
            "road_surface_condition": [surface_cond]
        }
        raw_df = pd.DataFrame(features_dict)

        # 5. Transform & Predict
        X_trans = self.preprocessor.transform(raw_df)
        probs = self.model.predict_proba(X_trans)[0]
        pred_idx = int(np.argmax(probs))
        pred_name = SEVERITY_NAMES[pred_idx]

        probs_dict = {
            "SLIGHT": round(float(probs[0]), 4),
            "SERIOUS": round(float(probs[1]), 4),
            "FATAL": round(float(probs[2]), 4)
        }

        # Elevated fatality risk alert (if fatal probability exceeds 10%)
        fatal_alert = (probs_dict["FATAL"] >= 0.10)

        latency_ms = round((time.time() - t0) * 1000, 2)
        geo_meta = self._check_geographic_coverage(req.latitude, req.longitude, req.timezone or "UTC")

        response = InjurySeverityPredictionResponse(
            predicted_severity=pred_name,
            severity_level_code=pred_idx,
            probabilities=probs_dict,
            fatality_risk_alert=fatal_alert,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            geographic_metadata=geo_meta
        )

        # 5. Database logging
        if db_session is not None:
            try:
                log_entry = InjuryPredictionLog(
                    id=response.prediction_id,
                    organization_id=organization_id,
                    latitude=req.latitude,
                    longitude=req.longitude,
                    predicted_severity=pred_name,
                    severity_level_code=pred_idx,
                    probabilities=probs_dict,
                    fatality_risk_alert=fatal_alert,
                    model_version=response.model_version,
                    latency_ms=latency_ms,
                    features_payload={k: v[0] for k, v in features_dict.items()}
                )
                db_session.add(log_entry)
                db_session.commit()
            except Exception as e:
                logger.error(f"Failed to log injury prediction to DB: {e}")
                db_session.rollback()

        return response

injury_severity_service = InjurySeverityService()
