@echo off
title SatQuery AI - Hackathon Backend Launcher
color 0b

echo ========================================================
echo    SatQuery AI - ST-09 Hackathon Backend Launcher
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "SatQuery AI - FastAPI Service" cmd /k "cd backend && .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo [2/2] Launching Cloudflare Public Tunnel ...
echo.
echo ========================================================
echo  LOOK BELOW FOR YOUR LIVE HTTPS URL:
echo  (Copy the https://...trycloudflare.com link)
echo ========================================================
echo.

.\cloudflared.exe tunnel --protocol http2 --url http://127.0.0.1:8000

pause
