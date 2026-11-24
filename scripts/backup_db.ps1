<#
.SYNOPSIS
    PostgreSQL Database Backup Script for Global Accident Intelligence SaaS (PowerShell)
.DESCRIPTION
    Creates a compressed backup of the PostgreSQL database and applies a 30-day retention policy.
.PARAMETER BackupDir
    Destination folder for database backups. Default: .\backups
#>

[CmdletBinding()]
param (
    [string]$BackupDir = ".\backups",
    [string]$ContainerName = "accident_postgres",
    [string]$DbUser = "postgres",
    [string]$DbName = "accident_db",
    [int]$RetentionDays = 30
)

$ErrorActionPreference = "Stop"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupFile = Join-Path $BackupDir "accident_db_$Timestamp.sql"

if (-not (Test-Path $BackupDir)) {
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting PostgreSQL Database Backup (PowerShell)" -ForegroundColor Cyan
Write-Host " Timestamp  : $Timestamp"
Write-Host " Database   : $DbName"
Write-Host " Container  : $ContainerName"
Write-Host " Destination: $BackupFile"
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if Docker container is available
$dockerRunning = $false
try {
    $containers = docker ps --format '{{.Names}}' 2>$null
    if ($containers -match "^$ContainerName$") {
        $dockerRunning = $true
    }
} catch {}

if ($dockerRunning) {
    Write-Host "[Backup] Dumping from Docker container '$ContainerName'..." -ForegroundColor Green
    docker exec -t $ContainerName pg_dump -U $DbUser -d $DbName --clean --if-exists --no-owner --no-privileges | Out-File -FilePath $BackupFile -Encoding utf8
} else {
    Write-Host "[Backup] Attempting host pg_dump command..." -ForegroundColor Yellow
    if (Get-Command pg_dump -ErrorAction SilentlyContinue) {
        pg_dump -h localhost -p 5432 -U $DbUser -d $DbName --clean --if-exists --no-owner --no-privileges | Out-File -FilePath $BackupFile -Encoding utf8
    } else {
        Write-Error "Neither running Docker container '$ContainerName' nor local 'pg_dump' utility was found."
        exit 1
    }
}

if ((Test-Path $BackupFile) -and ((Get-Item $BackupFile).Length -gt 0)) {
    $size = [math]::Round((Get-Item $BackupFile).Length / 1KB, 2)
    Write-Host "[Backup] Backup completed successfully. Size: $size KB" -ForegroundColor Green
} else {
    Write-Error "Backup file is empty or was not created."
    exit 1
}

# Retention policy
$cutoffDate = (Get-Date).AddDays(-$RetentionDays)
Get-ChildItem -Path $BackupDir -Filter "accident_db_*.sql" | Where-Object { $_.LastWriteTime -lt $cutoffDate } | Remove-Item -Force
Write-Host "[Backup] Retention policy applied (pruned backups older than $RetentionDays days)." -ForegroundColor Cyan
