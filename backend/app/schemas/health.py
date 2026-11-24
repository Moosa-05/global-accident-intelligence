from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    database: str
    models_ready: Dict[str, bool]
    model_loaded: bool
    model_version: Optional[str] = None
    timestamp: datetime = datetime.utcnow()
