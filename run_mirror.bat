@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"

if exist "%ROOT%\venv\Scripts\python.exe" (
    set "PY=%ROOT%\venv\Scripts\python.exe"
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PY=%ROOT%\.venv\Scripts\python.exe"
) else (
    echo [ERROR] No virtual environment found.
    echo Create one with:  python -m venv venv
    echo Then:             venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

echo [INFO] Starting FastAPI at http://127.0.0.1:8000 ...
start "VTO-API" /min cmd /c "cd /d "%ROOT%" && "%PY%" -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000 >> "%ROOT%\logs\app.log" 2>&1"

timeout /t 2 /nobreak >nul

echo [INFO] Launching VTO Smart Mirror...
"%PY%" "%ROOT%\main.py"
set "MIRROR_EXIT=%ERRORLEVEL%"

echo [INFO] Stopping FastAPI...
taskkill /FI "WINDOWTITLE eq VTO-API*" /T /F >nul 2>&1

endlocal & exit /b %MIRROR_EXIT%
