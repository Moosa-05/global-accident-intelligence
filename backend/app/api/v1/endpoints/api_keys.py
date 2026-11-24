"""API Key Management Endpoints for Telematics and Client Applications."""
from datetime import datetime, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.security import generate_api_key
from backend.app.db.models.api_key import ApiKey
from backend.app.db.models.user import User
from backend.app.schemas.api_key import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyItemResponse,
    ApiKeyRotateResponse,
    DEFAULT_SCOPES
)
from backend.app.api.deps import require_role

router = APIRouter()

@router.get("", response_model=List[ApiKeyItemResponse])
def list_organization_api_keys(
    user: User = Depends(require_role(["OWNER", "ADMIN"])),
    db: Session = Depends(get_db)
):
    """Lists all API keys for the current organization with prefix and usage metadata."""
    keys = db.query(ApiKey).filter(ApiKey.organization_id == user.organization_id).order_by(ApiKey.created_at.desc()).all()
    return keys

@router.post("", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    req: ApiKeyCreateRequest,
    user: User = Depends(require_role(["OWNER", "ADMIN"])),
    db: Session = Depends(get_db)
):
    """Creates a new organization-scoped API key. Full secret is returned ONLY ONCE."""
    full_key, key_prefix, hashed_secret = generate_api_key(is_live=req.is_live)
    
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)

    api_key_record = ApiKey(
        organization_id=user.organization_id,
        name=req.name,
        key_prefix=key_prefix,
        hashed_secret=hashed_secret,
        scopes=req.scopes or DEFAULT_SCOPES,
        is_active=True,
        expires_at=expires_at
    )
    db.add(api_key_record)
    db.commit()
    db.refresh(api_key_record)

    return ApiKeyCreateResponse(
        id=api_key_record.id,
        name=api_key_record.name,
        key_prefix=api_key_record.key_prefix,
        api_key=full_key,
        scopes=api_key_record.scopes,
        created_at=api_key_record.created_at,
        expires_at=api_key_record.expires_at
    )

@router.delete("/{key_id}", status_code=status.HTTP_200_OK)
def revoke_api_key(
    key_id: str,
    user: User = Depends(require_role(["OWNER", "ADMIN"])),
    db: Session = Depends(get_db)
):
    """Immediately revokes an API key, preventing any future inference requests."""
    key = db.query(ApiKey).filter(
        ApiKey.id == key_id,
        ApiKey.organization_id == user.organization_id
    ).first()
    if not key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.")

    if not key.is_active:
        return {"message": "API key is already revoked.", "key_id": key_id}

    key.is_active = False
    key.revoked_at = datetime.utcnow()
    db.commit()
    return {"message": "API key successfully revoked.", "key_id": key_id}

@router.post("/{key_id}/rotate", response_model=ApiKeyRotateResponse)
def rotate_api_key(
    key_id: str,
    user: User = Depends(require_role(["OWNER", "ADMIN"])),
    db: Session = Depends(get_db)
):
    """Rotates an API key: generates a new key with identical name and scopes, and revokes the old key."""
    old_key = db.query(ApiKey).filter(
        ApiKey.id == key_id,
        ApiKey.organization_id == user.organization_id
    ).first()
    if not old_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.")

    # 1. Revoke old key
    old_key.is_active = False
    old_key.revoked_at = datetime.utcnow()

    # 2. Generate new key
    full_key, key_prefix, hashed_secret = generate_api_key(is_live=old_key.key_prefix.startswith("agy_live"))
    new_key_record = ApiKey(
        organization_id=user.organization_id,
        name=f"{old_key.name} (Rotated)",
        key_prefix=key_prefix,
        hashed_secret=hashed_secret,
        scopes=old_key.scopes,
        is_active=True,
        expires_at=old_key.expires_at
    )
    db.add(new_key_record)
    db.commit()
    db.refresh(new_key_record)

    new_key_response = ApiKeyCreateResponse(
        id=new_key_record.id,
        name=new_key_record.name,
        key_prefix=new_key_record.key_prefix,
        api_key=full_key,
        scopes=new_key_record.scopes,
        created_at=new_key_record.created_at,
        expires_at=new_key_record.expires_at
    )

    return ApiKeyRotateResponse(
        message="API key rotated successfully. Previous key has been deactivated.",
        old_key_id=old_key.id,
        new_key=new_key_response
    )
