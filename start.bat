@echo off
echo Starting GPL Factory and Customer Runtime...
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found.
    echo Please run: python setup.py
    pause
    exit /b 1
)

echo Starting GPL Factory on port 8080...
start "GPL Factory" venv\Scripts\python.exe run.py

echo Starting GPL Customer Runtime on port 8081...
start "GPL Customer Runtime" venv\Scripts\python.exe run_customer.py

echo.
echo   Factory:          http://localhost:8080
echo   Customer Runtime: http://localhost:8081
echo.
echo Both processes started. Close their windows to stop them.
pause
