"""Core security module for SaaS authentication, JWT token generation,
password hashing, and API key cryptography.
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import bcrypt
from jose import jwt, JWTError

from backend.app.core.config import settings

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies plain password against standard bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:72],
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Generates standard salted bcrypt password hash."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8")[:72], salt).decode("utf-8")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

def hash_secret(secret: str) -> str:
    """Computes SHA-256 hash of API key secret."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()

def generate_api_key(is_live: bool = True) -> Tuple[str, str, str]:
    """Generates an organization-specific API key.
    Format: agy_live_<prefix_12_hex>_<secret_32_hex>
    Returns: (full_key, key_prefix, hashed_secret)
    """
    prefix_code = settings.API_KEY_PREFIX_LIVE if is_live else settings.API_KEY_PREFIX_TEST
    prefix_rnd = secrets.token_hex(6)  # 12 hex chars
    secret_rnd = secrets.token_hex(20) # 40 hex chars
    
    key_prefix = f"{prefix_code}{prefix_rnd}"
    full_key = f"{key_prefix}_{secret_rnd}"
    hashed_secret = hash_secret(secret_rnd)
    return full_key, key_prefix, hashed_secret

def verify_api_key(provided_key: str, key_prefix: str, hashed_secret: str) -> bool:
    """Verifies that a provided API key matches the stored prefix and hashed secret."""
    if not provided_key.startswith(f"{key_prefix}_"):
        return False
    parts = provided_key.split("_", 3)
    # agy, live, prefix, secret -> secret is the last part
    secret_part = provided_key[len(key_prefix) + 1:]
    return secrets.compare_digest(hash_secret(secret_part), hashed_secret)
