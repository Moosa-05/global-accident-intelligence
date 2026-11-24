import pytest
from unittest.mock import patch
from pydantic import ValidationError
from backend.app.core.config import Settings
from backend.app.db.session import get_engine

def test_production_rejects_insecure_default_secret_key():
    """In production mode, the default dev secret key MUST be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="accident-intelligence-super-secret-jwt-key-2026-global-production"
        )
    assert "Default SECRET_KEY cannot be used in production" in str(exc_info.value)

def test_production_rejects_short_secret_key():
    """In production mode, secret keys shorter than 32 characters MUST be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="too-short-secret-key"
        )
    assert "Production SECRET_KEY must be at least 32 characters" in str(exc_info.value)

def test_production_accepts_strong_secret_key():
    """In production mode, a strong secret key >= 32 chars is accepted."""
    s = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="a" * 32
    )
    assert s.ENVIRONMENT == "production"
    assert s.SECRET_KEY == "a" * 32

def test_cors_origins_parsing():
    """CORS origins should parse comma-delimited strings properly."""
    s = Settings(BACKEND_CORS_ORIGINS="https://app.example.com, https://api.example.com")
    assert s.cors_origins_list == ["https://app.example.com", "https://api.example.com"]

def test_production_strictly_forbids_silent_sqlite_fallback():
    """In production mode, failure to connect to PostgreSQL must raise RuntimeError and never silently fall back to SQLite."""
    from backend.app.core.config import settings
    with patch.object(settings, "ENVIRONMENT", "production"), \
         patch.object(settings, "DATABASE_URL", "postgresql+psycopg2://invalid_user:invalid_pass@127.0.0.1:5432/nonexistent_db"):
        with pytest.raises(RuntimeError) as exc_info:
            get_engine()
        assert "Silent fallback to SQLite is strictly forbidden in production mode" in str(exc_info.value)

