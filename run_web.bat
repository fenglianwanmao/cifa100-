@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=python"
where python >nul 2>&1
if errorlevel 1 (
    set "PY=py -3"
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found.
        pause
        exit /b 1
    )
)

echo Stopping old web process on port 5000 if any...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo Starting CIFAR-100 web demo...
echo Open http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.

%PY% cifar100_web.py
pause