@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
  echo.
  echo Start failed. Check .cache\logs\backend.log and .cache\logs\frontend.log
  pause
  exit /b 1
)
echo.
echo Browser: http://127.0.0.1:5173
start "" "http://127.0.0.1:5173"
pause
