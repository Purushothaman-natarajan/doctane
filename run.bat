@echo off
REM ================================================
REM Doctane - OCR Development Environment Launcher
REM ================================================

echo.
echo ========================================
echo    DOCTANE - Document Analysis System
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [INFO] Python detected

REM Create virtual environment if not exists
if not exist "venv" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [3/5] Installing dependencies...
pip install --upgrade pip -q 2>nul
pip install -q fastapi uvicorn python-multipart 2>nul
pip install -q -r requirements.txt 2>nul
pip install -q -e . 2>nul
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed. Continuing...
)

REM Check and create necessary directories
if not exist "data" mkdir data
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs

REM Start API server
echo [4/5] Starting API server...
echo.
echo Services will be available at:
echo   - API Server:    http://localhost:8000
echo   - API Docs:      http://localhost:8000/docs
echo   - Web Interface: http://localhost:8000/app
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python api\main.py

REM Deactivate on exit
deactivate