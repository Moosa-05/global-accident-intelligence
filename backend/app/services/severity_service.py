import time
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.units import UnitConverter, UnitSystem
from backend.app.adapters.weather_adapter import GenericWeatherAdapter
from backend.app.schemas.severity import SeverityPredictionRequest, SeverityPredictionResponse, GeographicMetadata
from backend.app.db.models.severity_prediction_log import SeverityPredictionLog

SEVERITY_DESCRIPTIONS = {
    1: "Minimal traffic delay / low road-network impact (typical duration < 15 mins).",
    2: "Moderate traffic slowdown / single-lane restriction (typical duration 15-45 mins).",
    3: "Significant traffic disruption / multi-lane or corridor delay (typical duration 45-90 mins).",
    4: "Severe road-network disruption / full corridor closure or major multi-vehicle incident."
}

SEVERITY_LEVEL_NAMES = {
    1: "MINIMAL_IMPACT",
    2: "MODERATE_IMPACT",
    3: "SIGNIFICANT_IMPACT",
    4: "SEVERE_DISRUPTION"
}

class SeverityPredictionService:
    def __init__(self):
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.model_version = "2.0.0"
        self.weather_adapter = GenericWeatherAdapter()
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            base_dir = Path(__file__).resolve().parents[3]
            model_path = base_dir / "models" / "accident_severity_lgbm.pkl"
            metadata_path = base_dir / "models" / "severity_model_metadata.json"
            schema_path = base_dir / "models" / "severity_feature_schema.json"

            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Loaded Severity LightGBM model from {model_path}")
            else:
                logger.warning(f"Severity model file not found at {model_path}")

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                    self.model_version = self.metadata.get("model_version", self.model_version)

            if schema_path.exists():
                with open(schema_path, "r") as f:
                    self.schema = json.load(f)
        except Exception as e:
            logger.error(f"Error loading severity model artifacts: {e}", exc_info=True)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _check_geographic_coverage(self, lat: float, lng: float) -> tuple[bool, str, Optional[str]]:
        """
        Geographic Out-Of-Distribution (OOD) detector.
        Checks whether coordinates lie within the empirical training distribution (US states).
        Does NOT reject requests; provides honest transparency regarding domain shift.
        """
        # Contiguous US bounds roughly: lat [24.5, 49.5], lng [-125.0, -66.9]
        # Alaska: lat [51.0, 71.5], lng [-179.0, -129.0]
        # Hawaii: lat [18.9, 22.3], lng [-160.5, -154.8]
        in_conus = (24.5 <= lat <= 49.5) and (-125.0 <= lng <= -66.9)
        in_alaska = (51.0 <= lat <= 71.5) and (-179.0 <= lng <= -129.0)
        in_hawaii = (18.9 <= lat <= 22.3) and (-160.5 <= lng <= -154.8)

        if in_conus or in_alaska or in_hawaii:
            return False, "EMPIRICALLY_VALIDATED", None

        is_ood = True
        status = "UNVALIDATED_TERRITORY"
        warning = (
            f"Location ({lat:.4f}, {lng:.4f}) is outside the empirical training coverage "
            "(US nationwide benchmark). Model predictions represent atmospheric and geometric "
            "heuristics without localized baseline calibration for this region."
        )
        return is_ood, status, warning

    def predict_severity(self, request: SeverityPredictionRequest, db_session = None, organization_id: Optional[str] = None) -> SeverityPredictionResponse:
        t0 = time.time()
        if not self.is_ready:
            raise RuntimeError("Severity ML model is not initialized.")

        # 1. Normalize units to canonical metric
        default_pres = 29.92 if request.units == UnitSystem.IMPERIAL else 1013.25
        pres = request.pressure if request.pressure is not None else default_pres
        default_temp = 68.0 if request.units == UnitSystem.IMPERIAL else 20.0
        default_wind = 8.0 if request.units == UnitSystem.IMPERIAL else 12.0
        default_vis = 10.0 if request.units == UnitSystem.IMPERIAL else 16.0

        norm_weather = UnitConverter.normalize_weather_inputs(
            temperature=request.temperature if request.temperature is not None else default_temp,
            wind_speed=request.wind_speed if request.wind_speed is not None else default_wind,
            visibility=request.visibility if request.visibility is not None else default_vis,
            pressure=pres,
            precipitation=request.precipitation or 0.0,
            input_units=request.units
        )

        weather_cat_code = self.weather_adapter.map_category_to_code(
            self.weather_adapter.map_condition(request.weather_condition or "Clear")
        )

        # 2. Compute temporal features
        ts = request.timestamp or datetime.utcnow()
        hour = ts.hour
        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
        is_night = int(hour < 6 or hour >= 20)

        # 3. Assemble canonical model feature vector
        feature_dict = {
            "latitude": float(request.latitude),
            "longitude": float(request.longitude),
            "temperature_c": float(norm_weather["temperature_c"]),
            "humidity_pct": float(request.humidity or 60.0),
            "pressure_hpa": float(norm_weather["pressure_hpa"]),
            "visibility_km": float(norm_weather["visibility_km"]),
            "wind_speed_kmh": float(norm_weather["wind_speed_kmh"]),
            "precipitation_mm": float(norm_weather["precipitation_mm"]),
            "weather_category": int(weather_cat_code),
            "is_night": int(is_night),
            "is_amenity": int(request.is_amenity),
            "is_crossing": int(request.is_crossing),
            "is_give_way": int(request.is_give_way),
            "is_junction": int(request.is_junction),
            "is_no_exit": int(request.is_no_exit),
            "is_railway": int(request.is_railway),
            "is_roundabout": int(request.is_roundabout),
            "is_station": int(request.is_station),
            "is_stop_sign": int(request.is_stop_sign),
            "is_traffic_calming": int(request.is_traffic_calming),
            "is_traffic_signal": int(request.is_traffic_signal),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos
        }

        df_in = pd.DataFrame([feature_dict])

        # 4. Predict class probabilities
        probas = self.model.predict_proba(df_in)[0]
        pred_idx = int(np.argmax(probas))
        pred_sev = pred_idx + 1

        proba_dict = {
            "class_1": round(float(probas[0]), 4),
            "class_2": round(float(probas[1]), 4),
            "class_3": round(float(probas[2]), 4),
            "class_4": round(float(probas[3]), 4)
        }

        latency_ms = round((time.time() - t0) * 1000, 2)

        # 5. Geographic OOD detection
        is_ood, val_status, val_warning = self._check_geographic_coverage(request.latitude, request.longitude)
        geo_meta = GeographicMetadata(
            coordinates={"latitude": request.latitude, "longitude": request.longitude},
            timezone=request.timezone,
            is_out_of_distribution=is_ood,
            validation_status=val_status,
            validation_warning=val_warning,
            training_geography="United States (49 States, 7.7M records benchmark)",
            licensing_terms="CC BY-NC-SA 4.0 Non-Commercial Research & Development Benchmark"
        )

        response = SeverityPredictionResponse(
            predicted_severity=pred_sev,
            severity_level=SEVERITY_LEVEL_NAMES[pred_sev],
            severity_probabilities=proba_dict,
            severity_description=SEVERITY_DESCRIPTIONS[pred_sev],
            model_version=self.model_version,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            geographic_metadata=geo_meta
        )

        # Log prediction to database if session provided
        if db_session is not None:
            try:
                log_entry = SeverityPredictionLog(
                    id=response.prediction_id,
                    organization_id=organization_id,
                    latitude=request.latitude,
                    longitude=request.longitude,
                    predicted_severity=pred_sev,
                    severity_level=response.severity_level,
                    probabilities=proba_dict,
                    model_version=self.model_version,
                    latency_ms=latency_ms,
                    features_payload=feature_dict
                )
                db_session.add(log_entry)
                db_session.commit()
            except Exception as e:
                logger.error(f"Failed to log severity prediction to DB: {e}")
                db_session.rollback()

        return response

severity_service = SeverityPredictionService()
