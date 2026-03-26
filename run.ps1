#!/usr/bin/env pwsh
# ================================================
# Doctane - OCR Development Environment Launcher
# ================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   DOCTANE - Document Analysis System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[INFO] Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+ from python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if not exists
if (-not (Test-Path "venv")) {
    Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip -q 2>$null
pip install -q fastapi uvicorn python-multipart 2>$null
pip install -q -r requirements.txt 2>$null
pip install -q -e . 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Some dependencies may have failed. Continuing..." -ForegroundColor Yellow
}

# Check and create necessary directories
if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }
if (-not (Test-Path "checkpoints")) { New-Item -ItemType Directory -Path "checkpoints" | Out-Null }
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

# Start API server
Write-Host "[4/5] Starting API server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Services will be available at:" -ForegroundColor Cyan
Write-Host "   - API Server:    http://localhost:8000" -ForegroundColor White
Write-Host "   - API Docs:      http://localhost:8000/docs" -ForegroundColor White
Write-Host "   - Web Interface: http://localhost:8000/app" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Start the server
python api\main.py

# Deactivate on exit
deactivate