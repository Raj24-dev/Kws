@echo off
rem Double-click to start the live dashboard: finds the board's USB port and opens http://localhost:8090.
rem Keep this window open while testing (closing it stops the dashboard). Extra arguments go to dashboard.py.
cd /d "%~dp0"
set "PY="
for /d %%D in ("C:\Espressif\tools\python\*") do if exist "%%D\venv\Scripts\python.exe" set "PY=%%D\venv\Scripts\python.exe"
if defined IDF_PYTHON_ENV_PATH set "PY=%IDF_PYTHON_ENV_PATH%\Scripts\python.exe"
if not defined PY set "PY=python"
"%PY%" dashboard.py %*
pause
