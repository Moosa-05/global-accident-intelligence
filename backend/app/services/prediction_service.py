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
from backend.app.schemas.prediction import RiskPredictionRequest, RiskPredictionResponse
from backend.app.db.models.prediction_log import PredictionLog

class RiskPredictionService:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.threshold = settings.DEFAULT_RISK_THRESHOLD
        self.model_version = "1.0.0"
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            model_path = Path(settings.MODEL_PATH)
            pipeline_path = Path(settings.PIPELINE_PATH)
            metadata_path = Path(settings.METADATA_PATH)
            schema_path = Path(settings.SCHEMA_PATH)

            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Loaded LightGBM model from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}")

            if pipeline_path.exists():
                self.pipeline = joblib.load(pipeline_path)
                logger.info(f"Loaded Feature Pipeline from {pipeline_path}")
            else:
                logger.warning(f"Pipeline file not found at {pipeline_path}")

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                    self.threshold = float(self.metadata.get("optimal_threshold", self.threshold))
                    self.model_version = self.metadata.get("model_version", self.model_version)
                    logger.info(f"Loaded metadata. Operational threshold set to {self.threshold:.4f}")

            if schema_path.exists():
                with open(schema_path, "r") as f:
                    self.schema = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.pipeline is not None

    def _determine_risk_level(self, prob: float) -> str:
        if prob < 0.03:
            return "LOW"
        elif prob < self.threshold:
            return "MODERATE"
        elif prob < 0.20:
            return "HIGH"
        else:
            return "CRITICAL"

    def predict_risk(self, request: RiskPredictionRequest, db_session = None, organization_id: Optional[str] = None) -> RiskPredictionResponse:
        t0 = time.time()
        
        # 1. Determine hour and cyclic features
        hour = request.hour if request.hour is not None else datetime.utcnow().hour
        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))

        # 2. Canonicalize road corridor identifier
        raw_road = str(request.road or "00I24E").strip().upper()
        if raw_road in ["I24E", "00I24E", "I-24E", "I-24 EAST"]:
            canonical_road = "00I24E"
        elif raw_road in ["I24W", "00I24W", "I-24W", "I-24 WEST"]:
            canonical_road = "00I24W"
        elif raw_road in ["I75N", "00I75N", "I-75N", "I-75 NORTH"]:
            canonical_road = "00I75N"
        elif raw_road in ["I75S", "00I75S", "I-75S", "I-75 SOUTH"]:
            canonical_road = "00I75S"
        elif raw_road in ["SR29N", "0SR29N", "US27N", "US-27N"]:
            canonical_road = "0SR29N"
        elif raw_road in ["SR29S", "0SR29S", "US27S", "US-27S"]:
            canonical_road = "0SR29S"
        elif raw_road in ["SR153N", "153N"]:
            canonical_road = "SR153N"
        elif raw_road in ["SR153S", "153S"]:
            canonical_road = "SR153S"
        else:
            canonical_road = raw_road

        # 3. Canonicalize weather condition
        raw_weather = str(request.weather or "Clear").strip().lower()
        if any(w in raw_weather for w in ["rain", "drizzle", "wet", "storm", "fog", "mist"]):
            canonical_weather = "Rain"
        elif any(w in raw_weather for w in ["snow", "sleet", "hail", "ice", "blizzard"]):
            canonical_weather = "Snow"
        else:
            canonical_weather = "Clear"

        # 4. Canonicalize lighting condition
        raw_light = str(request.light or "Daylight").strip().lower()
        if any(l in raw_light for l in ["dark", "night"]):
            canonical_light = "Dark"
        elif "dawn" in raw_light:
            canonical_light = "Dawn"
        elif "dusk" in raw_light or "twilight" in raw_light:
            canonical_light = "Dusk"
        else:
            canonical_light = "Daylight"

        # 5. Handle speed drop (deceleration shockwave)
        # In traffic engineering, speed drop is colloquially reported as a positive magnitude
        # of deceleration (e.g., 20 mph drop), whereas the underlying LightGBM pipeline
        # expects signed speed delta (last 5 min avg - first 5 min avg <= 0).
        # We ensure positive deceleration magnitudes map to negative speed changes.
        raw_drop = float(request.speed_drop or 0.0)
        signed_speed_drop = -abs(raw_drop) if raw_drop != 0.0 else 0.0

        # Handle upstream / downstream defaults if missing
        speed_up = request.speed_upstream if request.speed_upstream is not None else request.speed_mean
        vol_up = request.volume_upstream if request.volume_upstream is not None else request.volume_mean
        occ_up = request.occupancy_upstream if request.occupancy_upstream is not None else request.occupancy_mean

        speed_dn = request.speed_downstream if request.speed_downstream is not None else request.speed_mean
        vol_dn = request.volume_downstream if request.volume_downstream is not None else request.volume_mean
        occ_dn = request.occupancy_downstream if request.occupancy_downstream is not None else request.occupancy_mean

        speed_grad = float(speed_dn - speed_up)
        occ_grad = float(occ_up - occ_dn)
        congestion = float(request.occupancy_mean / (request.speed_mean + 1.0))

        # 6. Realistic speed bounds accounting for shockwaves
        if request.speed_min is not None:
            speed_min = request.speed_min
        else:
            base_min = request.speed_mean - (request.speed_std or 0.0)
            if signed_speed_drop < 0:
                base_min = min(base_min, request.speed_mean + signed_speed_drop)
            speed_min = max(0.0, base_min)

        speed_max = request.speed_max if request.speed_max is not None else min(120.0, request.speed_mean + (request.speed_std or 0.0))

        feature_dict = {
            'road': canonical_road,
            'mile': request.mile,
            'hour': hour,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'weather': canonical_weather,
            'light': canonical_light,
            'speed_mean': request.speed_mean,
            'speed_std': request.speed_std or 0.0,
            'speed_min': speed_min,
            'speed_max': speed_max,
            'speed_drop': signed_speed_drop,
            'volume_mean': request.volume_mean,
            'volume_std': request.volume_std or 0.0,
            'occupancy_mean': request.occupancy_mean,
            'occupancy_max': request.occupancy_max if request.occupancy_max is not None else request.occupancy_mean + (request.occupancy_std or 0.0),
            'occupancy_std': request.occupancy_std or 0.0,
            'speed_upstream': speed_up,
            'volume_upstream': vol_up,
            'occupancy_upstream': occ_up,
            'speed_downstream': speed_dn,
            'volume_downstream': vol_dn,
            'occupancy_downstream': occ_dn,
            'speed_gradient': speed_grad,
            'occupancy_gradient': occ_grad,
            'congestion_index': congestion
        }

        # Build single row dataframe for pipeline
        df_in = pd.DataFrame([feature_dict])

        if not self.is_ready:
            raise RuntimeError("ML model or preprocessing pipeline is not initialized.")

        X_proc = self.pipeline.transform(df_in)
        proba = float(self.model.predict_proba(X_proc)[0, 1])
        alert = bool(proba >= self.threshold)
        level = self._determine_risk_level(proba)
        latency_ms = round((time.time() - t0) * 1000, 2)

        response = RiskPredictionResponse(
            road=canonical_road,
            mile=request.mile,
            risk_score=round(proba, 4),
            risk_level=level,
            accident_alert=alert,
            threshold_used=round(self.threshold, 4),
            model_version=self.model_version,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat()
        )

        # Log prediction to database if session is provided
        if db_session is not None:
            try:
                log_entry = PredictionLog(
                    id=response.prediction_id,
                    organization_id=organization_id,
                    road=response.road,
                    mile=response.mile,
                    risk_score=response.risk_score,
                    risk_level=response.risk_level,
                    accident_alert=response.accident_alert,
                    threshold_used=response.threshold_used,
                    model_version=response.model_version,
                    latency_ms=response.latency_ms,
                    features_payload=feature_dict
                )
                db_session.add(log_entry)
                db_session.commit()
            except Exception as e:
                logger.error(f"Failed to log prediction to DB: {e}")
                db_session.rollback()

        return response

prediction_service = RiskPredictionService()
