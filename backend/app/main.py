from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, logger
from backend.app.db.base import Base
from backend.app.db.session import engine
from backend.app.api.v1.api import api_router
from backend.app.api.v1.endpoints.health import router as health_router

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT.upper()}]")
    # Initialize DB tables (for dev/testing fallback; in production alembic migrations are executed)
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schemas verified successfully.")
    except Exception as e:
        logger.error(f"Failed to verify database schemas: {e}")
    yield
    logger.info("Graceful shutdown completed.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

# Trusted Host Middleware (when configured and not wildcard)
allowed_hosts = settings.allowed_hosts_list
if allowed_hosts and allowed_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# Restrictive Production CORS
cors_origins = settings.cors_origins_list
has_wildcard = "*" in cors_origins

if has_wildcard:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With", "X-API-Key", "X-Session-ID"],
        expose_headers=["X-Session-ID"],
    )

# Root-level health check (/health, /health/live, /health/ready)
app.include_router(health_router, prefix="", tags=["Root Health"])

# Versioned API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
