<#
.SYNOPSIS
    PostgreSQL Database Restore Script for Global Accident Intelligence SaaS (PowerShell)
.DESCRIPTION
    Restores the PostgreSQL database from a SQL backup file with safety confirmation.
.PARAMETER BackupFile
    Path to the .sql backup file to restore.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory=$true)]
    [string]$BackupFile,
    [string]$ContainerName = "accident_postgres",
    [string]$DbUser = "postgres",
    [string]$DbName = "accident_db"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $BackupFile)) {
    Write-Error "Backup file '$BackupFile' does not exist."
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Red
Write-Host " PostgreSQL Database Restore Pre-Flight Check" -ForegroundColor Yellow
Write-Host " Backup File: $BackupFile"
Write-Host " Database   : $DbName"
Write-Host " Container  : $ContainerName"
Write-Host "==========================================================" -ForegroundColor Red
Write-Warning "This operation will overwrite data in '$DbName' with the contents of the backup file."

$confirmation = Read-Host "Type 'RESTORE' to proceed"
if ($confirmation -ne "RESTORE") {
    Write-Host "Restore operation cancelled." -ForegroundColor Cyan
    exit 0
}

Write-Host "[Restore] Restoring database..." -ForegroundColor Green

$dockerRunning = $false
try {
    $containers = docker ps --format '{{.Names}}' 2>$null
    if ($containers -match "^$ContainerName$") {
        $dockerRunning = $true
    }
} catch {}

if ($dockerRunning) {
    Write-Host "[Restore] Streaming to Docker container '$ContainerName'..." -ForegroundColor Green
    Get-Content $BackupFile -Raw | docker exec -i $ContainerName psql -U $DbUser -d $DbName
} else {
    if (Get-Command psql -ErrorAction SilentlyContinue) {
        Get-Content $BackupFile -Raw | psql -h localhost -p 5432 -U $DbUser -d $DbName
    } else {
        Write-Error "Neither running Docker container '$ContainerName' nor local 'psql' utility was found."
        exit 1
    }
}

Write-Host "[Restore] Database restore completed successfully." -ForegroundColor Green
