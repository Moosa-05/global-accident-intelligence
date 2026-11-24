"""Authentication Schemas for Registration, Login, and Tokens."""
from typing import Optional, List
from pydantic import BaseModel, Field

class UserRegisterRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$", description="Valid email address")
    password: str = Field(..., min_length=8, description="Password minimum 8 characters")
    full_name: Optional[str] = None
    organization_name: str = Field(..., min_length=2, description="Organization name")
    subscription_tier: Optional[str] = Field("FREE", description="Initial plan: FREE, DEVELOPER, PROFESSIONAL, etc.")
    research_access_requested: Optional[bool] = Field(False, description="Whether organization requests academic research access")

class UserLoginRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$", description="Valid email address")
    password: str

class TokenRefreshRequest(BaseModel):
    refresh_token: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    organization_id: str

class AuthContext(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    organization_id: str
    organization_name: str
    subscription_tier: str
    is_commercial_cleared: bool
    research_access_permitted: bool
    scopes: List[str] = []
    auth_method: str = "jwt"  # jwt or api_key
    api_key_id: Optional[str] = None
