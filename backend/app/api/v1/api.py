from fastapi import APIRouter
from backend.app.api.v1.endpoints import (
    health,
    auth,
    organizations,
    api_keys,
    subscriptions,
    predictions,
    severity,
    combined,
    injury,
    exposure,
    models,
    history
)

api_router = APIRouter()

# Health endpoints
api_router.include_router(health.router, tags=["Health"])

# Multi-Tenant Auth & Account Management
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["Organizations"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["API Keys"])
api_router.include_router(subscriptions.router, prefix="/subscriptions", tags=["Subscriptions"])

# Prediction endpoints
api_router.include_router(predictions.router, prefix="/predictions", tags=["Risk Predictions"])
api_router.include_router(severity.router, prefix="/predictions", tags=["Severity Predictions"])
api_router.include_router(combined.router, prefix="/predictions", tags=["Combined Predictions"])
api_router.include_router(injury.router, prefix="/predictions", tags=["Human Injury Severity"])
api_router.include_router(exposure.router, prefix="/predictions", tags=["Road Exposure Risk"])

# Prediction history & audit logs
api_router.include_router(history.router, prefix="/history", tags=["Prediction History"])

# Model registry & metadata
api_router.include_router(models.router, tags=["Models Registry"])
