@echo off
REM Starts the Resume Completeness Assessment web demo and opens the browser.
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" src\14_web_app.py
pause
