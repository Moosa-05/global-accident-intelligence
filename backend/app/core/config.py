from typing import Optional, List, Union
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field, model_validator
import logging

# Root path of the repository
BASE_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "Global Accident Intelligence"
    VERSION: str = "2.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DEMO_MODE: bool = True

    # Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "accident_db"
    DATABASE_URL: Optional[str] = None
    SQLITE_FALLBACK_URL: str = f"sqlite:///{BASE_DIR / 'accident_dev.db'}"

    # Database Connection Pool Settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800

    @computed_field
    def sync_database_uri(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # SaaS Security & Auth Settings
    SECRET_KEY: str = "accident-intelligence-super-secret-jwt-key-2026-global-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    API_KEY_PREFIX_LIVE: str = "agy_live_"
    API_KEY_PREFIX_TEST: str = "agy_test_"

    # SaaS Rate Limiting & Enforcement
    ENABLE_RATE_LIMITING: bool = True
    DEFAULT_RATE_LIMIT_PER_MINUTE: int = 60
    ENFORCE_MODEL_LICENSING: bool = True

    # CORS & Trusted Hosts
    BACKEND_CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
    ALLOWED_HOSTS: Union[List[str], str] = ["*"]

    # External Monitoring & Observability
    SENTRY_DSN: Optional[str] = None

    # ML & Model Serving Paths
    MODEL_PATH: str = str(BASE_DIR / "models" / "accident_risk_lgbm.pkl")
    PIPELINE_PATH: str = str(BASE_DIR / "models" / "feature_pipeline.pkl")
    METADATA_PATH: str = str(BASE_DIR / "models" / "model_metadata.json")
    SCHEMA_PATH: str = str(BASE_DIR / "models" / "feature_schema.json")
    DEFAULT_RISK_THRESHOLD: float = 0.30
    MLFLOW_TRACKING_URI: str = str(BASE_DIR / "mlruns")

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",") if origin.strip()]
        return self.BACKEND_CORS_ORIGINS

    @property
    def allowed_hosts_list(self) -> List[str]:
        if isinstance(self.ALLOWED_HOSTS, str):
            return [host.strip() for host in self.ALLOWED_HOSTS.split(",") if host.strip()]
        return self.ALLOWED_HOSTS

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.ENVIRONMENT.lower() == "production":
            # 1. Validate Secret Key
            default_insecure_key = "accident-intelligence-super-secret-jwt-key-2026-global-production"
            if self.SECRET_KEY == default_insecure_key:
                raise ValueError(
                    "CRITICAL SECURITY VIOLATION: Default SECRET_KEY cannot be used in production. "
                    "Generate a cryptographically secure key (e.g., `openssl rand -hex 32`) and set SECRET_KEY."
                )
            if len(self.SECRET_KEY) < 32:
                raise ValueError("CRITICAL SECURITY VIOLATION: Production SECRET_KEY must be at least 32 characters long.")

            # 2. Validate CORS Configuration
            origins = self.cors_origins_list
            if "*" in origins:
                logging.getLogger("uvicorn.error").warning(
                    "SECURITY WARNING: Production CORS contains wildcard '*'. "
                    "Configure BACKEND_CORS_ORIGINS with specific trusted client domains."
                )
        return self

settings = Settings()
