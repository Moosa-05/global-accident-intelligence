#!/bin/bash
# ==============================================================================
# Database Backup Script for Global Accident Intelligence SaaS (PostgreSQL)
# ==============================================================================
# Usage:
#   ./scripts/backup_db.sh [backup_directory]
# Example:
#   ./scripts/backup_db.sh /var/backups/accident
# ==============================================================================

set -euo pipefail

BACKUP_DIR="${1:-./backups}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/accident_db_${TIMESTAMP}.sql.gz"
CONTAINER_NAME="${DB_CONTAINER:-accident_postgres}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${POSTGRES_DB:-accident_db}"
RETENTION_DAYS=30

mkdir -p "${BACKUP_DIR}"

echo "=========================================================="
echo " Starting PostgreSQL Database Backup"
echo " Timestamp : ${TIMESTAMP}"
echo " Database  : ${DB_NAME}"
echo " Container : ${CONTAINER_NAME}"
echo " Destination: ${BACKUP_FILE}"
echo "=========================================================="

# Check if Docker container is running or use local pg_dump
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[Backup] Dumping from Docker container '${CONTAINER_NAME}'..."
    docker exec -t "${CONTAINER_NAME}" pg_dump -U "${DB_USER}" -d "${DB_NAME}" --clean --if-exists --no-owner --no-privileges | gzip > "${BACKUP_FILE}"
elif command -v pg_dump >/dev/null 2>&1; then
    echo "[Backup] Dumping using host pg_dump CLI..."
    pg_dump -h "${POSTGRES_SERVER:-localhost}" -p "${POSTGRES_PORT:-5432}" -U "${DB_USER}" -d "${DB_NAME}" --clean --if-exists --no-owner --no-privileges | gzip > "${BACKUP_FILE}"
else
    echo "ERROR: Neither running Docker container '${CONTAINER_NAME}' nor local 'pg_dump' was found." >&2
    exit 1
fi

# Verify backup file size
if [ -s "${BACKUP_FILE}" ]; then
    SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
    echo "[Backup] Backup completed successfully. Size: ${SIZE}"
else
    echo "ERROR: Backup file was created but is empty." >&2
    exit 1
fi

# Clean up older backups according to retention policy
echo "[Backup] Pruning backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "accident_db_*.sql.gz" -type f -mtime +"${RETENTION_DAYS}" -delete
echo "[Backup] Retention policy applied."
echo "=========================================================="
