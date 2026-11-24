"""Licensing Enforcement Engine for SaaS Prediction Endpoints.
Ensures commercial accounts cannot invoke non-commercial models (US Accidents CC BY-NC-SA 4.0
or Chattanooga Research Benchmark), and validates dual licensing on composite routes.
"""
from typing import Dict, Any, Optional
from fastapi import HTTPException, status
from backend.app.schemas.auth import AuthContext

MODEL_LICENSING_SPECS: Dict[str, Dict[str, Any]] = {
    "accident_risk_v1": {
        "model_id": "accident_risk_v1",
        "category": "RESEARCH_BENCHMARK",
        "license_name": "Academic Research Benchmark (Tennessee DOT Corridor)",
        "commercial_use_permitted": False,
        "research_access_only": True,
        "provenance": "TDOT Chattanooga Freeway Sensor Network (18,059 events)",
        "timescale": "Micro-Scale Immediate (15-Minute Corridor Event Window)",
        "target_definition": "Binary accident occurrence probability in [0, 1] conditional on real-time 30-second loop sensor telemetry.",
        "commercial_alternative": "For commercial road safety assessment, use '/api/v1/predictions/exposure-risk' (UK/France Open Government Data)."
    },
    "accident_severity_v2": {
        "model_id": "accident_severity_v2",
        "category": "RESTRICTED_NON_COMMERCIAL",
        "license_name": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        "commercial_use_permitted": False,
        "research_access_only": True,
        "provenance": "US Accidents March 2023 Release (Moosavi et al., 7.728M records)",
        "timescale": "Post-Collision Network Disruption Window (15 to 120+ minutes)",
        "target_definition": "US DoT traffic delay disruption severity (Classes 1 through 4); not physiological human injury trauma.",
        "commercial_alternative": "For commercially cleared physiological trauma severity, use '/api/v1/predictions/injury-severity' (UK STATS19 & France BAAC)."
    },
    "human_injury_severity_v1": {
        "model_id": "human_injury_severity_v1",
        "category": "COMMERCIAL_CLEARED",
        "license_name": "Open Government Licence v3.0 (UK) & Licence Ouverte v2.0 (France)",
        "commercial_use_permitted": True,
        "research_access_only": False,
        "provenance": "UK Department for Transport STATS19 (2022) & France ONISR BAAC (2023) [160,804 events]",
        "timescale": "Immediate Collision Trauma Outcome (UNECE / WHO 30-Day Physiological Standard)",
        "target_definition": "3-class human physical injury severity: SLIGHT, SERIOUS, FATAL.",
        "commercial_alternative": None
    },
    "exposure_risk_v1": {
        "model_id": "exposure_risk_v1",
        "category": "COMMERCIAL_CLEARED",
        "license_name": "Open Government Licence v3.0 (UK) & Licence Ouverte v2.0 (France)",
        "commercial_use_permitted": True,
        "research_access_only": False,
        "provenance": "UK DfT AADF Traffic Counts (17,835 segments) & France TMJA National Network (1,098 segments)",
        "timescale": "Macro-Scale Annual Road Segment Planning Horizon",
        "target_definition": "Expected annual collision frequency E[Y] and normalized crash rate per Million Vehicle-Kilometres Travelled (MVKT).",
        "commercial_alternative": None
    }
}

def verify_model_licensing_access(model_id: str, auth_context: Optional[AuthContext]) -> Dict[str, Any]:
    """Enforces commercial licensing restrictions at the individual model/prediction endpoint level.
    If auth_context is None (e.g. unauthenticated demo/test mode), allows execution if not strictly enforced.
    If commercial organization attempts to invoke non-commercial models without research permission, raises 403 Forbidden.
    """
    spec = MODEL_LICENSING_SPECS.get(model_id)
    if not spec:
        return {}

    # If auth context provided, enforce policy
    if auth_context is not None:
        # Check if model is restricted to non-commercial / research
        if not spec["commercial_use_permitted"]:
            # If the organization is in commercial mode and has NO research permit:
            if auth_context.is_commercial_cleared and not auth_context.research_access_permitted:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error_code": "COMMERCIAL_LICENSING_RESTRICTION",
                        "model_id": model_id,
                        "license_type": spec["license_name"],
                        "message": (
                            f"Model '{model_id}' is restricted to non-commercial research use under {spec['license_name']}. "
                            f"Commercial SaaS subscription ({auth_context.subscription_tier}) is prohibited from invoking non-commercial models."
                        ),
                        "suggested_alternative": spec["commercial_alternative"]
                    }
                )
    return spec

def verify_combined_licensing_access(auth_context: Optional[AuthContext]) -> None:
    """Verifies that the caller has legal entitlement to invoke the combined prediction route,
    which combines 'accident_risk_v1' and 'accident_severity_v2'.
    """
    if auth_context is not None:
        if auth_context.is_commercial_cleared and not auth_context.research_access_permitted:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "COMBINED_ROUTE_LICENSING_RESTRICTION",
                    "models_involved": ["accident_risk_v1 (Research Benchmark)", "accident_severity_v2 (CC BY-NC-SA 4.0)"],
                    "message": (
                        "The combined prediction endpoint incorporates experimental non-commercial research models: "
                        "'accident_severity_v2' (licensed under CC BY-NC-SA 4.0 Non-Commercial) and 'accident_risk_v1' (academic sensor benchmark). "
                        "Commercial SaaS tenants cannot execute this endpoint without an Academic/Research dual-licensing addendum."
                    ),
                    "commercial_alternatives": [
                        {"endpoint": "/api/v1/predictions/injury-severity", "license": "OGL v3.0 / Licence Ouverte v2.0 (Commercial Cleared)"},
                        {"endpoint": "/api/v1/predictions/exposure-risk", "license": "OGL v3.0 / Licence Ouverte v2.0 (Commercial Cleared)"}
                    ]
                }
            )
