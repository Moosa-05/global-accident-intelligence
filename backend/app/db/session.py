from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.base import Base

def get_engine():
    primary_uri = settings.sync_database_uri
    is_postgres = "postgresql" in primary_uri
    is_production = settings.ENVIRONMENT.lower() == "production"

    try:
        engine_kwargs = {"pool_pre_ping": True}
        if is_postgres:
            engine_kwargs.update({
                "connect_args": {"connect_timeout": 5},
                "pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW,
                "pool_timeout": settings.DB_POOL_TIMEOUT,
                "pool_recycle": settings.DB_POOL_RECYCLE,
            })

        engine = create_engine(primary_uri, **engine_kwargs)

        # Test connection
        with engine.connect() as conn:
            pass
        logger.info(f"Connected to primary database: {primary_uri.split('@')[-1] if '@' in primary_uri else primary_uri}")
        return engine
    except Exception as e:
        if is_production:
            error_msg = (
                f"CRITICAL PRODUCTION ERROR: Failed to connect to PostgreSQL ({primary_uri}): {e}. "
                "Silent fallback to SQLite is strictly forbidden in production mode."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        logger.warning(f"Could not connect to primary database ({primary_uri}): {e}. Falling back to SQLite.")
        fallback_engine = create_engine(
            settings.SQLITE_FALLBACK_URL,
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=fallback_engine)
        return fallback_engine

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
