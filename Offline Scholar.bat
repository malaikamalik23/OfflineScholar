@echo off
rem Double-click to open Offline Scholar. Uses the project's .venv (see README: Desktop app).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The .venv folder is missing. Follow the README setup steps first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" scripts\desktop.py
