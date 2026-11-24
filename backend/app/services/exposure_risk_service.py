"""Exposure-Adjusted Road Segment Crash Frequency & Rate Prediction Service.
Uses fitted Negative Binomial GLM with log(MVKT) exposure offset.
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
from backend.app.schemas.exposure_risk import (
    ExposureRiskPredictionRequest,
    ExposureRiskPredictionResponse
)
from backend.app.db.models.exposure_risk_prediction_log import ExposureRiskPredictionLog

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NATIONAL_MOTORWAY_BASELINE_RATE = 0.055  # ~0.055 reported injury collisions per million vehicle-km on European motorways

class ExposureRiskService:
    def __init__(self):
        self.model = None
        self.model_type = "glm"
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        models_dir = PROJECT_ROOT / "models"
        lgb_path = models_dir / "exposure_risk_lgbm.pkl"
        nb_path = models_dir / "exposure_risk_nb.pkl"
        meta_path = models_dir / "exposure_model_metadata.json"

        try:
            if meta_path.exists():
                import json
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)

            if lgb_path.exists():
                self.model = joblib.load(lgb_path)
                self.model_type = "lightgbm"
                logger.info("Exposure Risk Promoted LightGBM Poisson model loaded successfully.")
            elif nb_path.exists():
                self.model = joblib.load(nb_path)
                self.model_type = "glm"
                logger.info("Exposure Risk Negative Binomial GLM loaded successfully.")
            else:
                logger.warning(f"No exposure risk model found at {lgb_path} or {nb_path}.")
        except Exception as e:
            logger.error(f"Failed to load exposure risk model: {e}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def model_version(self) -> str:
        return "2.1.0" if self.model_type == "lightgbm" else "1.0.0"

    def predict(self, req: ExposureRiskPredictionRequest, db_session = None, organization_id: Optional[str] = None) -> ExposureRiskPredictionResponse:
        t0 = time.time()

        # 1. Calculate Annual Million Vehicle-Kilometres Travelled (MVKT)
        # MVKT = AADF * 365 * length_km / 1,000,000
        mvkt = (float(req.annual_average_daily_flow) * 365.0 * float(req.link_length_km)) / 1e6
        mvkt = max(mvkt, 0.001)

        # 2. Road category mapping
        road_cls = req.road_class.upper()
        if road_cls not in ["MOTORWAY", "TRUNK_HIGHWAY", "PRIMARY_ARTERIAL", "SECONDARY_COLLECTOR", "LOCAL_OTHER"]:
            road_cls = "PRIMARY_ARTERIAL"

        # 3. Predict expected annual crash count
        if self.model_type == "lightgbm":
            df_lgb = pd.DataFrame([{
                "road_class": road_cls,
                "speed_limit_kmh": float(req.speed_limit_kmh),
                "link_length_km": float(req.link_length_km),
                "all_motor_vehicles": float(req.annual_average_daily_flow),
                "latitude": float(req.latitude),
                "longitude": float(req.longitude)
            }])
            df_lgb["road_class"] = pd.Categorical(df_lgb["road_class"], categories=['MOTORWAY', 'PRIMARY_ARTERIAL', 'TRUNK_HIGHWAY'])
            pred_rate = float(self.model.predict(df_lgb)[0])
            pred_crashes = pred_rate * mvkt
        else:
            df_input = pd.DataFrame([{
                "road_class": road_cls,
                "speed_limit_kmh": float(req.speed_limit_kmh),
                "latitude": float(req.latitude),
                "longitude": float(req.longitude),
                "mvkt": mvkt,
                "log_mvkt": math.log(mvkt)
            }])
            pred_crashes = float(self.model.predict(df_input, offset=df_input["log_mvkt"])[0])

        pred_crashes = max(pred_crashes, 0.0)

        # 4. Normalized crash rate per MVKT
        crash_rate_per_mvkt = round(pred_crashes / mvkt, 4)
        relative_risk = round(crash_rate_per_mvkt / NATIONAL_MOTORWAY_BASELINE_RATE, 2)

        # 5. Risk Tier Assignment
        if crash_rate_per_mvkt < 0.20:
            risk_tier = "LOW_RISK"
        elif crash_rate_per_mvkt < 0.80:
            risk_tier = "TYPICAL_RISK"
        elif crash_rate_per_mvkt < 2.00:
            risk_tier = "ELEVATED_RISK"
        else:
            risk_tier = "HIGH_RISK"

        latency_ms = round((time.time() - t0) * 1000, 2)

        response = ExposureRiskPredictionResponse(
            annual_exposure_million_vkt=round(mvkt, 4),
            predicted_annual_crashes=round(pred_crashes, 2),
            predicted_crash_rate_per_million_vkt=crash_rate_per_mvkt,
            risk_tier=risk_tier,
            relative_risk_to_national_baseline=relative_risk,
            statistical_model="Negative Binomial GLM / LightGBM Poisson with Log-Exposure Offset" if self.model_type == "lightgbm" else "Negative Binomial GLM with Log-Exposure Offset",
            model_version=self.model_version,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat()
        )

        # 6. Database logging
        if db_session is not None:
            try:
                log_entry = ExposureRiskPredictionLog(
                    id=response.prediction_id,
                    organization_id=organization_id,
                    road_name=req.road_name,
                    road_class=road_cls,
                    latitude=req.latitude,
                    longitude=req.longitude,
                    annual_average_daily_flow=req.annual_average_daily_flow,
                    link_length_km=req.link_length_km,
                    annual_exposure_million_vkt=response.annual_exposure_million_vkt,
                    predicted_annual_crashes=response.predicted_annual_crashes,
                    predicted_crash_rate_per_million_vkt=response.predicted_crash_rate_per_million_vkt,
                    risk_tier=response.risk_tier,
                    model_version=response.model_version,
                    latency_ms=latency_ms
                )
                db_session.add(log_entry)
                db_session.commit()
            except Exception as e:
                logger.error(f"Failed to log exposure risk prediction to DB: {e}")
                db_session.rollback()

        return response

exposure_risk_service = ExposureRiskService()
