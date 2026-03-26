@echo off
REM Doctane Launcher - Starts both Backend API and Frontend

echo.
echo ========================================
echo    DOCTANE - Document Analysis System
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [3/4] Installing dependencies...
pip install -q fastapi uvicorn python-multipart 2>nul
pip install -q -r requirements.txt 2>nul
pip install -q -e . 2>nul

REM Start API server in background
echo [4/4] Starting services...
echo.
echo Starting API server on http://localhost:8000
echo Opening web interface...
echo.

REM Start API server
start "Doctane API" cmd /k "python api\main.py"

REM Open browser after a short delay
timeout /t 3 /nobreak >nul
start http://localhost:8000/app.html

echo.
echo Services started:
echo   - API:    http://localhost:8000
echo   - Docs:   http://localhost:8000/docs
echo   - Frontend: http://localhost:8000/app.html
echo.
echo Press any key to stop the API server...
pause

taskkill /F /IM python.exe /FI "WINDOWTITLE eq Doctane API*" 2>nul
deactivate