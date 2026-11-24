from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from backend.app.schemas.prediction import RiskPredictionRequest, RiskPredictionResponse
from backend.app.schemas.severity import SeverityPredictionRequest, SeverityPredictionResponse, GeographicMetadata

class CombinedPredictionRequest(BaseModel):
    risk_inputs: RiskPredictionRequest = Field(..., description="Corridor sensor telemetry for accident occurrence risk estimation")
    severity_inputs: SeverityPredictionRequest = Field(..., description="Atmospheric and road infrastructure context for severity assessment")

class RiskSummary(BaseModel):
    risk_score: float
    risk_level: str
    accident_alert: bool
    threshold_used: float

class SeveritySummary(BaseModel):
    predicted_severity: int
    severity_level: str
    severity_probabilities: Dict[str, float]
    definition: str = "Traffic Impact Delay Severity (US DoT benchmark standard; conditional on accident occurrence; not physiological injury/fatality scale)"

class CombinedPredictionResponse(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    accident_risk: RiskSummary
    conditional_severity: SeveritySummary
    expected_severity_index: float = Field(..., description="Expected traffic severity value on [1.0, 4.0] scale: sum(c * P(c))")
    joint_risk_impact_score: float = Field(..., description="Composite risk-weighted impact: P(accident) * (expected_severity / 4.0) on [0.0, 1.0]")
    model_versions: Dict[str, str]
    latency_ms: float
    timestamp: str
    geographic_metadata: Optional[GeographicMetadata] = None

