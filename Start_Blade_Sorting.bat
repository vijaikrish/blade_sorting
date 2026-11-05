@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo The project environment is missing. Follow README.md setup first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
if errorlevel 1 pause
