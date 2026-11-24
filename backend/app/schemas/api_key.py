"""API Key Schemas for Scoped Permissions and Rotation."""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

DEFAULT_SCOPES = [
    "predict:risk",
    "predict:severity",
    "predict:combined",
    "predict:injury",
    "predict:exposure",
    "models:read"
]

class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Friendly key label, e.g., 'Fleet Tracking Production'")
    scopes: Optional[List[str]] = Field(default_factory=lambda: DEFAULT_SCOPES)
    is_live: bool = Field(True, description="True for agy_live_, False for agy_test_")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)

class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    api_key: str = Field(..., description="Full secret key. Displayed ONLY ONCE!")
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime]

class ApiKeyItemResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: List[str]
    is_active: bool
    total_requests: int
    last_used_at: Optional[datetime]
    created_at: datetime
    expires_at: Optional[datetime]
    revoked_at: Optional[datetime]

class ApiKeyRotateResponse(BaseModel):
    message: str = "API key successfully rotated. Old key revoked."
    old_key_id: str
    new_key: ApiKeyCreateResponse
