#!/bin/bash
# ==============================================================================
# Database Restore Script for Global Accident Intelligence SaaS (PostgreSQL)
# ==============================================================================
# Usage:
#   ./scripts/restore_db.sh <path_to_backup_file>
# Example:
#   ./scripts/restore_db.sh backups/accident_db_20261119_120000.sql.gz
# ==============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_backup_file>" >&2
    exit 1
fi

BACKUP_FILE="$1"
CONTAINER_NAME="${DB_CONTAINER:-accident_postgres}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${POSTGRES_DB:-accident_db}"

if [ ! -f "${BACKUP_FILE}" ]; then
    echo "ERROR: Backup file '${BACKUP_FILE}' does not exist." >&2
    exit 1
fi

echo "=========================================================="
echo " PostgreSQL Database Restore Pre-Flight Check"
echo " Backup File: ${BACKUP_FILE}"
echo " Database   : ${DB_NAME}"
echo " Container  : ${CONTAINER_NAME}"
echo "=========================================================="
echo "WARNING: This will overwrite tables in '${DB_NAME}' with data from the backup."
read -p "Type 'RESTORE' to confirm: " CONFIRMATION

if [ "${CONFIRMATION}" != "RESTORE" ]; then
    echo "Restore cancelled by user."
    exit 0
fi

echo "[Restore] Restoring database..."
if [[ "${BACKUP_FILE}" == *.gz ]]; then
    DECOMPRESS_CMD="gunzip -c"
else
    DECOMPRESS_CMD="cat"
fi

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[Restore] Streaming to Docker container '${CONTAINER_NAME}'..."
    ${DECOMPRESS_CMD} "${BACKUP_FILE}" | docker exec -i "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}"
elif command -v psql >/dev/null 2>&1; then
    echo "[Restore] Streaming using host psql..."
    ${DECOMPRESS_CMD} "${BACKUP_FILE}" | psql -h "${POSTGRES_SERVER:-localhost}" -p "${POSTGRES_PORT:-5432}" -U "${DB_USER}" -d "${DB_NAME}"
else
    echo "ERROR: Neither running Docker container '${CONTAINER_NAME}' nor local 'psql' was found." >&2
    exit 1
fi

echo "[Restore] Database restore completed successfully."
echo "=========================================================="
