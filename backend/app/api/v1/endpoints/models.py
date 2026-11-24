from fastapi import APIRouter, Depends
from backend.app.schemas.model_info import ModelsListResponse, ModelMetadataItem
from backend.app.schemas.auth import AuthContext
from backend.app.api.deps import (
    get_prediction_service,
    get_severity_service,
    get_injury_severity_service,
    get_exposure_risk_service,
    get_auth_context
)
from backend.app.services.prediction_service import RiskPredictionService
from backend.app.services.severity_service import SeverityPredictionService
from backend.app.services.injury_severity_service import InjurySeverityService
from backend.app.services.exposure_risk_service import ExposureRiskService

router = APIRouter()

@router.get("/models", response_model=ModelsListResponse)
def get_available_models(
    commercial_only: bool = False,
    risk_svc: RiskPredictionService = Depends(get_prediction_service),
    sev_svc: SeverityPredictionService = Depends(get_severity_service),
    inj_svc: InjurySeverityService = Depends(get_injury_severity_service),
    exp_svc: ExposureRiskService = Depends(get_exposure_risk_service),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    """
    Returns registry information, tasks, training geographies, and licensing status
    for all machine learning models in the SaaS platform.
    Supports commercial licensing enforcement via `commercial_only=True`.
    """
    risk_meta = risk_svc.metadata
    sev_meta = sev_svc.metadata
    inj_meta = inj_svc.metadata or {}
    exp_meta = exp_svc.metadata or {}

    models_list = [
        ModelMetadataItem(
            model_id="accident_risk_v1",
            model_name="Accident Risk Prediction Model (Micro-Scale)",
            model_version=risk_svc.model_version,
            task_description="Estimates real-time probability of an accident occurring in the next 15 minutes along a monitored freeway corridor.",
            target_type="Binary Classification (0: No Incident, 1: Incident Occurred)",
            status="READY" if risk_svc.is_ready else "UNAVAILABLE",
            commercial_use_permitted=False,
            licensing_category="RESEARCH_BENCHMARK",
            training_data_source="TDOT Chattanooga Freeway Sensor Network (18,059 events)",
            training_geography="United States (Chattanooga, TN I-24 / I-75 / SR-153 sensor corridors)",
            coverage_and_scope_notes="Experimental sensor-dependent model. Requires 11-station inductive loop or radar speed/occupancy corridor telemetry; not suited for arbitrary geographic coordinates lacking telemetry.",
            licensing={
                "status": "Academic Research Benchmark",
                "notes": "Trained on public Tennessee DOT sensor telemetry. Non-commercial experimental benchmark."
            },
            metrics=risk_meta.get("test_metrics", {})
        ),
        ModelMetadataItem(
            model_id="accident_severity_v2",
            model_name="Accident Traffic-Impact Severity Model",
            model_version=sev_svc.model_version,
            task_description="Estimates 4-class traffic impact delay severity (Classes 1 through 4) conditional on an accident occurring.",
            target_type="Multiclass Classification (1: Minimal, 2: Moderate, 3: Significant, 4: Severe Disruption)",
            status="READY" if sev_svc.is_ready else "UNAVAILABLE",
            commercial_use_permitted=False,
            licensing_category="RESTRICTED_NON_COMMERCIAL",
            training_data_source="US Accidents March 2023 Release (Moosavi et al., 7.728M records)",
            training_geography="United States (Nationwide, 49 states + DC)",
            coverage_and_scope_notes="Country-independent canonical schema accepting global WGS84 coordinates, IANA time zones, and metric/imperial weather. Out-of-distribution detection flags international coordinates.",
            licensing={
                "dataset_license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
                "commercial_status": "RESTRICTED. Non-Commercial Research & Pre-Commercial SaaS Validation Only."
            },
            metrics=sev_meta.get("test_metrics", {})
        ),
        ModelMetadataItem(
            model_id="human_injury_severity_v1",
            model_name="Human Physiological Injury Severity Model",
            model_version=inj_svc.model_version,
            task_description="Predicts physical trauma severity (SLIGHT, SERIOUS, FATAL) using UNECE/WHO 30-day standard definitions.",
            target_type="3-Class Classification (SLIGHT, SERIOUS, FATAL)",
            status="READY" if inj_svc.is_ready else "UNAVAILABLE",
            commercial_use_permitted=True,
            licensing_category="COMMERCIAL_CLEARED",
            training_data_source="UK Department for Transport STATS19 (2022) & France ONISR BAAC (2023) [160,804 events]",
            training_geography="United Kingdom & France (Western Europe)",
            coverage_and_scope_notes="Fully country-independent schema with WGS84 coordinates and ERA5 reanalysis weather. Validated across multi-country holdouts, spatial holdouts, and cross-country transfer.",
            licensing={
                "dataset_license": "Open Government Licence v3.0 (UK) & Licence Ouverte v2.0 (France)",
                "commercial_status": "COMMERCIAL SAAS CLEARED. Perpetual, royalty-free commercial exploitation authorized."
            },
            metrics=inj_meta.get("evaluation_metrics", {}).get("test_multi_country", {})
        ),
        ModelMetadataItem(
            model_id="exposure_risk_v1",
            model_name="Road Segment Annual Crash Frequency & Exposure Risk Model",
            model_version=exp_svc.model_version,
            task_description="Estimates expected annual injury crash frequency E[Y] and normalized crash rate per Million Vehicle-Kilometres Travelled (MVKT).",
            target_type="Poisson / LightGBM Count & Rate Regression with log-exposure offset",
            status="READY" if exp_svc.is_ready else "UNAVAILABLE",
            commercial_use_permitted=True,
            licensing_category="COMMERCIAL_CLEARED",
            training_data_source="UK DfT AADF Traffic Counts (17,835 links) & France TMJA National Network (1,098 links)",
            training_geography="Great Britain & France National Road Network",
            coverage_and_scope_notes="Macro-scale annual planning model based on verified traffic flow exposure. Strictly separate from instantaneous micro-scale hourly congestion risk.",
            licensing={
                "dataset_license": "Open Government Licence v3.0 (UK) & Licence Ouverte v2.0 (France)",
                "commercial_status": "COMMERCIAL SAAS CLEARED. Perpetual, royalty-free commercial exploitation authorized."
            },
            metrics=exp_meta.get("benchmark_results", {})
        )
    ]

    total_count = len(models_list)
    comm_count = sum(1 for m in models_list if m.commercial_use_permitted)
    restr_count = total_count - comm_count

    if commercial_only:
        filtered_models = [m for m in models_list if m.commercial_use_permitted]
    else:
        filtered_models = models_list

    return ModelsListResponse(
        platform_name="Global Accident Intelligence Platform",
        api_version="2.1.0",
        total_registered_models=total_count,
        commercial_cleared_count=comm_count,
        restricted_research_count=restr_count,
        global_extensibility={
            "coordinate_system": "WGS84 Latitude / Longitude (Global)",
            "supported_units": ["metric", "imperial"],
            "timezone_standard": "IANA Timezone Database (tzdata)",
            "weather_normalization": "Autonomous unit normalization via UnitConverter + Copernicus ERA5 Reanalysis",
            "provider_adapters": ["GenericWeatherAdapter", "TelematicsAdapter", "RoadGeometryAdapter", "UKStats19Adapter", "FranceBAACAdapter", "ERA5WeatherAdapter"]
        },
        models=filtered_models
    )
