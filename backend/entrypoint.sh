#!/bin/sh
set -e

echo "=========================================================="
echo " Starting Global Accident Intelligence SaaS Backend"
echo " Environment: ${ENVIRONMENT:-production}"
echo "=========================================================="

# If PostgreSQL is configured, verify database connectivity and apply migrations
if [ "$ENVIRONMENT" = "production" ] || [ -n "$DATABASE_URL" ]; then
    echo "[Entrypoint] Verifying database connectivity..."
    python - <<'EOF'
import sys, time
from sqlalchemy import create_engine, text
from backend.app.core.config import settings

max_retries = 30
retry_interval = 2
uri = settings.sync_database_uri

print(f"[Entrypoint] Connecting to: {uri.split('@')[-1] if '@' in uri else uri}")
for attempt in range(1, max_retries + 1):
    try:
        engine = create_engine(uri, connect_args={"connect_timeout": 3} if "postgresql" in uri else {})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[Entrypoint] Database connected successfully on attempt {attempt}.")
        sys.exit(0)
    except Exception as e:
        print(f"[Entrypoint] Database not ready yet (attempt {attempt}/{max_retries}): {e}")
        time.sleep(retry_interval)

print("[Entrypoint] FATAL: Database connection timed out after 60 seconds.")
sys.exit(1)
EOF

    echo "[Entrypoint] Running Alembic database migrations..."
    alembic -c backend/alembic.ini upgrade head
    echo "[Entrypoint] Migrations completed successfully."
fi

# Launch Uvicorn with configurable worker processes
WORKERS=${WEB_CONCURRENCY:-4}
echo "[Entrypoint] Starting Uvicorn with $WORKERS worker processes..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers "$WORKERS"
